import sys
import time
import serial
import serial.tools.list_ports
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QLineEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QGroupBox, QGridLayout, QCheckBox,
                             QMessageBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QPalette
import pyqtgraph as pg


class SerialWorker(QThread):
    """
    Worker thread to handle serial communication without blocking the UI
    """
    # Signal to send data back to the main thread
    dataReceived = pyqtSignal(list)
    connectionStatus = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.is_running = False
        self.channels = 10  # Number of channels to receive data for

    def connect_serial(self, port, baud_rate):
        """Connect to the serial port"""
        try:
            self.serial_port = serial.Serial(port, baud_rate, timeout=1)
            self.is_running = True
            self.connectionStatus.emit(True)
            return True
        except Exception as e:
            print(f"Serial connection error: {e}")
            self.connectionStatus.emit(False)
            return False

    def disconnect_serial(self):
        """Disconnect from the serial port"""
        self.is_running = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.connectionStatus.emit(False)

    def run(self):
        """Main thread loop for reading serial data"""
        buffer = bytearray()

        while self.is_running:
            if self.serial_port and self.serial_port.is_open:
                try:
                    # Check if there's data to read
                    if self.serial_port.in_waiting > 0:
                        data = self.serial_port.read(self.serial_port.in_waiting)
                        buffer.extend(data)

                        # Based on the format provided: "0B EB 0B E1 0B E2 0B E4 0B E1 0B E0 0B DE 0B E6 01 4A 01 5A 31"
                        # It appears to be 10 channels with 2 bytes each (except for the last byte which might be a checksum)
                        # Each channel seems to be represented by 2 bytes in hexadecimal format

                        # Check if we have enough bytes for a complete message
                        # 10 channels * 2 bytes per channel = 20 bytes + 1 potential checksum byte = 21 bytes
                        required_bytes = 21

                        while len(buffer) >= required_bytes:
                            # Extract channel values (2 bytes per channel)
                            channel_values = []

                            for i in range(self.channels):
                                # Extract 2 bytes and convert to an integer value
                                # Each channel is represented by 2 bytes in big-endian order
                                start_index = i * 2
                                high_byte = buffer[start_index]
                                low_byte = buffer[start_index + 1]
                                value = (high_byte << 8) | low_byte
                                channel_values.append(value)

                            # Emit the data to be processed by the main thread
                            self.dataReceived.emit(channel_values)

                            # Remove processed data from buffer
                            buffer = buffer[required_bytes:]

                            # Print received data for debugging
                            hex_values = ' '.join([f"{val:04X}" for val in channel_values])
                            print(f"Received data: {hex_values}")

                except Exception as e:
                    print(f"Error reading serial data: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.1)


class AerationDetectionSystem(QMainWindow):
    """
    Main application window for the Water Flow Aeration Detection System
    """

    def __init__(self):
        super().__init__()

        # Window properties
        self.setWindowTitle("水流掺气检测系统")
        self.resize(1200, 700)

        # Data storage
        self.channel_count = 10
        self.raw_values = [0] * self.channel_count
        self.zero_points = [0] * self.channel_count
        self.full_points = [3000] * self.channel_count
        self.aeration_rates = [0.0] * self.channel_count

        # Graph data storage
        self.graph_time_data = []
        self.graph_data = [[] for _ in range(self.channel_count)]
        self.max_points = 100  # Maximum points to display on graph

        # Set up the serial worker thread
        self.serial_worker = SerialWorker()
        self.serial_worker.dataReceived.connect(self.process_data)
        self.serial_worker.connectionStatus.connect(self.update_connection_status)

        # Set up the UI
        self.setup_ui()

        # Set up update timer for the UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)  # Update UI every 100ms

        # Logging flag
        self.logging_enabled = False
        self.log_file = None

    def setup_ui(self):
        """Set up the user interface"""
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Control panel section
        control_group = QGroupBox("控制面板")
        control_layout = QGridLayout()

        # Serial port selection
        control_layout.addWidget(QLabel("串口:"), 0, 0)
        self.port_selector = QComboBox()
        self.refresh_ports()
        control_layout.addWidget(self.port_selector, 0, 1)

        # Baud rate selection
        control_layout.addWidget(QLabel("波特率:"), 0, 2)
        self.baud_selector = QComboBox()
        self.baud_selector.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baud_selector.setCurrentText("115200")
        control_layout.addWidget(self.baud_selector, 0, 3)

        # Connect button
        self.connect_button = QPushButton("连接")
        self.connect_button.clicked.connect(self.toggle_connection)
        control_layout.addWidget(self.connect_button, 0, 4)

        # Status indicator
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: red")
        control_layout.addWidget(self.status_indicator, 0, 5)

        # Reset button
        self.reset_button = QPushButton("刷新串口")
        self.reset_button.clicked.connect(self.refresh_ports)
        control_layout.addWidget(self.reset_button, 0, 6)

        # Logging control
        self.log_button = QPushButton("开始记录")
        self.log_button.clicked.connect(self.toggle_logging)
        control_layout.addWidget(self.log_button, 1, 0, 1, 2)

        # Performance mode button
        self.perf_button = QPushButton("进入演示模式")
        self.perf_button.clicked.connect(self.toggle_performance_mode)
        control_layout.addWidget(self.perf_button, 1, 2, 1, 2)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # Calibration section
        calib_group = QGroupBox("校准设置")
        calib_layout = QGridLayout()

        # Channel selection
        calib_layout.addWidget(QLabel("通道:"), 0, 0)
        self.channel_selector = QComboBox()
        for i in range(1, self.channel_count + 1):
            self.channel_selector.addItem(f"{i}")
        calib_layout.addWidget(self.channel_selector, 0, 1)

        # Zero point setting
        calib_layout.addWidget(QLabel("零点值:"), 0, 2)
        self.zero_input = QLineEdit("0")
        calib_layout.addWidget(self.zero_input, 0, 3)

        # Full point setting
        calib_layout.addWidget(QLabel("满点值:"), 0, 4)
        self.full_input = QLineEdit("3000")
        calib_layout.addWidget(self.full_input, 0, 5)

        # Calibration buttons
        self.set_calib_button = QPushButton("设置校准")
        self.set_calib_button.clicked.connect(self.set_calibration)
        calib_layout.addWidget(self.set_calib_button, 0, 6)

        self.get_zero_button = QPushButton("获取当前值为零点")
        self.get_zero_button.clicked.connect(self.get_current_as_zero)
        calib_layout.addWidget(self.get_zero_button, 1, 2, 1, 2)

        self.get_full_button = QPushButton("获取当前值为满点")
        self.get_full_button.clicked.connect(self.get_current_as_full)
        calib_layout.addWidget(self.get_full_button, 1, 4, 1, 2)

        calib_group.setLayout(calib_layout)
        main_layout.addWidget(calib_group)

        # Data display and graph section
        data_graph_layout = QHBoxLayout()

        # Data table
        data_display_group = QGroupBox("数据显示")
        data_layout = QVBoxLayout()

        # Table with editable cells
        self.data_table = QTableWidget(self.channel_count, 5)
        self.data_table.setHorizontalHeaderLabels(["通道", "原始值", "零点", "满点", "掺气率(%)"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Help label for table editing
        help_label = QLabel("提示：双击零点或满点单元格可以直接修改校准值")
        help_label.setStyleSheet("color: blue;")
        data_layout.addWidget(help_label)

        # Initialize table with channel data
        for i in range(self.channel_count):
            # Channel number (with colored indicator)
            channel_item = QTableWidgetItem(f"通道 {i + 1}")
            channel_item.setBackground(self.get_channel_color(i))
            self.data_table.setItem(i, 0, channel_item)

            # Raw value (read-only)
            raw_item = QTableWidgetItem("0")
            raw_item.setFlags(raw_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
            self.data_table.setItem(i, 1, raw_item)

            # Zero point (editable)
            zero_item = QTableWidgetItem("0")
            self.data_table.setItem(i, 2, zero_item)

            # Full point (editable)
            full_item = QTableWidgetItem("3000")
            self.data_table.setItem(i, 3, full_item)

            # Aeration rate (read-only)
            rate_item = QTableWidgetItem("0.00")
            rate_item.setFlags(rate_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
            self.data_table.setItem(i, 4, rate_item)

        # Connect cell change signal
        self.data_table.itemChanged.connect(self.on_table_cell_changed)

        data_layout.addWidget(self.data_table)
        data_display_group.setLayout(data_layout)
        data_graph_layout.addWidget(data_display_group)

        # Graph section
        graph_group = QGroupBox("掺气率实时曲线")
        graph_layout = QVBoxLayout()

        # Create graph widget
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('w')
        self.graph_widget.setLabel('left', '掺气率 (%)')
        self.graph_widget.setLabel('bottom', '时间 (秒)')
        self.graph_widget.showGrid(x=True, y=True)
        self.graph_widget.setYRange(0, 100)

        # Create plot lines for each channel
        self.plot_lines = []
        for i in range(self.channel_count):
            pen = pg.mkPen(color=self.get_channel_color(i, qt_color=True), width=2)
            line = self.graph_widget.plot([], [], pen=pen, name=f"通道 {i + 1}")
            self.plot_lines.append(line)

        graph_layout.addWidget(self.graph_widget)

        # Channel visibility controls
        channel_controls = QHBoxLayout()
        channel_controls.addWidget(QLabel("显示通道:"))

        self.channel_checkboxes = []
        for i in range(self.channel_count):
            cb = QCheckBox(f"{i + 1}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_graph_visibility)
            cb.setStyleSheet(f"QCheckBox {{ color: {self.get_channel_color(i, css=True)}; }}")
            self.channel_checkboxes.append(cb)
            channel_controls.addWidget(cb)

        # Select all/none buttons
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(lambda: self.set_all_channels(True))
        channel_controls.addWidget(select_all_btn)

        select_none_btn = QPushButton("取消")
        select_none_btn.clicked.connect(lambda: self.set_all_channels(False))
        channel_controls.addWidget(select_none_btn)

        graph_layout.addLayout(channel_controls)
        graph_group.setLayout(graph_layout)
        data_graph_layout.addWidget(graph_group)

        main_layout.addLayout(data_graph_layout)

        # Status bar
        self.statusBar().showMessage("发现 0 个串口设备")

        self.setCentralWidget(central_widget)

    def on_table_cell_changed(self, item):
        """Handle changes in the table cells"""
        row = item.row()
        column = item.column()

        # Only handle zero point and full point columns (2 and 3)
        if column not in [2, 3]:
            return

        try:
            value = int(item.text())

            # Get the current zero and full points for this row
            current_zero = self.zero_points[row]
            current_full = self.full_points[row]

            # Update the appropriate value
            if column == 2:  # Zero point
                new_zero = value
                new_full = current_full
                # Make sure zero is less than full
                if new_zero >= current_full:
                    QMessageBox.warning(self, "校准错误", f"零点值 ({new_zero}) 必须小于满点值 ({current_full})！")
                    # Revert to previous value
                    item.setText(str(current_zero))
                    return
                self.zero_points[row] = new_zero

            elif column == 3:  # Full point
                new_zero = current_zero
                new_full = value
                # Make sure full is greater than zero
                if new_full <= current_zero:
                    QMessageBox.warning(self, "校准错误", f"满点值 ({new_full}) 必须大于零点值 ({current_zero})！")
                    # Revert to previous value
                    item.setText(str(current_full))
                    return
                self.full_points[row] = new_full

            # If this is the currently selected channel in the calibration panel,
            # update the input fields there as well
            if row == self.channel_selector.currentIndex():
                if column == 2:  # Zero point
                    self.zero_input.setText(str(new_zero))
                elif column == 3:  # Full point
                    self.full_input.setText(str(new_full))

            self.statusBar().showMessage(f"通道 {row + 1} 校准参数已更新")

        except ValueError:
            # Restore the previous value if the entered text is not a valid integer
            if column == 2:
                item.setText(str(self.zero_points[row]))
            elif column == 3:
                item.setText(str(self.full_points[row]))
            QMessageBox.warning(self, "输入错误", "请输入有效的整数值！")

    def refresh_ports(self):
        """Refresh the list of available serial ports"""
        self.port_selector.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_selector.addItem(port.device)

        self.statusBar().showMessage(f"发现 {len(ports)} 个串口设备")

    def toggle_connection(self):
        """Connect or disconnect from the serial port"""
        if not self.serial_worker.is_running:
            port = self.port_selector.currentText()
            baud_rate = int(self.baud_selector.currentText())

            if not port:
                self.statusBar().showMessage("错误: 未选择串口")
                return

            # Start the worker thread
            if self.serial_worker.connect_serial(port, baud_rate):
                self.serial_worker.start()
                self.connect_button.setText("断开连接")
            else:
                self.statusBar().showMessage(f"无法连接到 {port}")
        else:
            # Stop the worker thread
            self.serial_worker.disconnect_serial()
            self.connect_button.setText("连接")

    def update_connection_status(self, connected):
        """Update the connection status indicator"""
        if connected:
            self.status_indicator.setStyleSheet("color: green")
            self.statusBar().showMessage("已连接")
        else:
            self.status_indicator.setStyleSheet("color: red")
            self.statusBar().showMessage("未连接")

    def process_data(self, data):
        """Process the received data from serial port"""
        if len(data) == self.channel_count:
            self.raw_values = data

            # Calculate aeration rates
            for i in range(self.channel_count):
                # Avoid division by zero
                full_range = self.full_points[i] - self.zero_points[i]
                if full_range != 0:
                    normalized_value = (data[i] - self.zero_points[i]) / full_range
                    # Ensure value is within 0-100% range
                    self.aeration_rates[i] = max(0, min(100, normalized_value * 100))
                else:
                    self.aeration_rates[i] = 0.0

            # Log data if enabled
            if self.logging_enabled and self.log_file:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                log_line = f"{timestamp},"
                log_line += ",".join([str(val) for val in self.raw_values])
                log_line += "," + ",".join([f"{rate:.2f}" for rate in self.aeration_rates])
                self.log_file.write(log_line + "\n")
                self.log_file.flush()

            # Add data to graph
            current_time = time.time()
            if not self.graph_time_data:
                # First data point, use 0 as reference
                self.graph_start_time = current_time
                self.graph_time_data.append(0)
            else:
                self.graph_time_data.append(current_time - self.graph_start_time)

            # Add data to each channel
            for i in range(self.channel_count):
                self.graph_data[i].append(self.aeration_rates[i])

            # Limit the number of points displayed
            if len(self.graph_time_data) > self.max_points:
                self.graph_time_data.pop(0)
                for i in range(self.channel_count):
                    self.graph_data[i].pop(0)

    def update_ui(self):
        """Update the UI with current values"""
        # Temporarily disconnect the itemChanged signal to prevent triggering callbacks
        self.data_table.itemChanged.disconnect(self.on_table_cell_changed)

        for i in range(self.channel_count):
            self.data_table.item(i, 1).setText(str(self.raw_values[i]))
            self.data_table.item(i, 2).setText(str(self.zero_points[i]))
            self.data_table.item(i, 3).setText(str(self.full_points[i]))
            self.data_table.item(i, 4).setText(f"{self.aeration_rates[i]:.2f}")

        # Reconnect the signal
        self.data_table.itemChanged.connect(self.on_table_cell_changed)

        # Update graph
        if self.graph_time_data:
            for i in range(self.channel_count):
                if self.channel_checkboxes[i].isChecked():
                    self.plot_lines[i].setData(self.graph_time_data, self.graph_data[i])

        # Update status bar with current date/time
        current_status = self.statusBar().currentMessage()
        if not current_status.startswith("发现") and not current_status.startswith("错误"):
            self.statusBar().showMessage(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")

    def set_calibration(self):
        """Set calibration values for the selected channel"""
        try:
            channel_idx = self.channel_selector.currentIndex()
            zero_value = int(self.zero_input.text())
            full_value = int(self.full_input.text())

            # Validation
            if full_value <= zero_value:
                self.statusBar().showMessage("错误: 满点值必须大于零点值")
                return

            self.zero_points[channel_idx] = zero_value
            self.full_points[channel_idx] = full_value

            # Update table (this will not trigger the itemChanged signal because we'll disconnect it first)
            self.data_table.itemChanged.disconnect(self.on_table_cell_changed)
            self.data_table.item(channel_idx, 2).setText(str(zero_value))
            self.data_table.item(channel_idx, 3).setText(str(full_value))
            self.data_table.itemChanged.connect(self.on_table_cell_changed)

            self.statusBar().showMessage(f"通道 {channel_idx + 1} 校准设置已应用")
        except ValueError:
            self.statusBar().showMessage("错误: 请输入有效的数字")

    def get_current_as_zero(self):
        """Set current value as zero point for selected channel"""
        channel_idx = self.channel_selector.currentIndex()
        current_value = self.raw_values[channel_idx]

        # Validate against full point
        if current_value >= self.full_points[channel_idx]:
            QMessageBox.warning(self, "校准错误",
                                f"当前值 ({current_value}) 大于等于满点值 ({self.full_points[channel_idx]})！")
            return

        self.zero_points[channel_idx] = current_value
        self.zero_input.setText(str(current_value))

        # Update table
        self.data_table.itemChanged.disconnect(self.on_table_cell_changed)
        self.data_table.item(channel_idx, 2).setText(str(current_value))
        self.data_table.itemChanged.connect(self.on_table_cell_changed)

        self.statusBar().showMessage(f"通道 {channel_idx + 1} 零点值已设置为当前值 {current_value}")

    def get_current_as_full(self):
        """Set current value as full point for selected channel"""
        channel_idx = self.channel_selector.currentIndex()
        current_value = self.raw_values[channel_idx]

        # Validate against zero point
        if current_value <= self.zero_points[channel_idx]:
            QMessageBox.warning(self, "校准错误",
                                f"当前值 ({current_value}) 小于等于零点值 ({self.zero_points[channel_idx]})！")
            return

        self.full_points[channel_idx] = current_value
        self.full_input.setText(str(current_value))

        # Update table
        self.data_table.itemChanged.disconnect(self.on_table_cell_changed)
        self.data_table.item(channel_idx, 3).setText(str(current_value))
        self.data_table.itemChanged.connect(self.on_table_cell_changed)

        self.statusBar().showMessage(f"通道 {channel_idx + 1} 满点值已设置为当前值 {current_value}")

    def toggle_logging(self):
        """Toggle data logging on/off"""
        if not self.logging_enabled:
            try:
                log_filename = f"aeration_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                self.log_file = open(log_filename, 'w')

                # Write header
                header = "Timestamp,"
                header += ",".join([f"Raw_{i + 1}" for i in range(self.channel_count)])
                header += "," + ",".join([f"Rate_{i + 1}" for i in range(self.channel_count)])
                self.log_file.write(header + "\n")

                self.logging_enabled = True
                self.log_button.setText("停止记录")
                self.statusBar().showMessage(f"开始记录数据到 {log_filename}")
            except Exception as e:
                self.statusBar().showMessage(f"无法创建日志文件: {e}")
        else:
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            self.logging_enabled = False
            self.log_button.setText("开始记录")
            self.statusBar().showMessage("数据记录已停止")

    def toggle_performance_mode(self):
        """Toggle performance mode (generate simulated data)"""
        if hasattr(self, 'perf_mode_timer') and self.perf_mode_timer.isActive():
            # Stop performance mode
            self.perf_mode_timer.stop()
            self.perf_button.setText("进入演示模式")
            self.status_indicator.setStyleSheet("color: red")
            self.statusBar().showMessage("演示模式已停止")
        elif not self.serial_worker.is_running:
            # Start generating simulated data
            self.perf_mode_timer = QTimer()
            self.perf_mode_timer.timeout.connect(self.generate_test_data)
            self.perf_mode_timer.start(100)  # Generate data every 100ms
            self.perf_button.setText("退出演示模式")
            self.status_indicator.setStyleSheet("color: blue")
            self.statusBar().showMessage("演示模式已启动")
        else:
            self.statusBar().showMessage("请先断开串口连接以启用演示模式")

    def generate_test_data(self):
        """Generate simulated data for testing"""
        # Create sine waves with different phases for each channel
        time_val = time.time()
        test_data = []
        for i in range(self.channel_count):
            # Create some variation between channels
            phase = i * 0.5
            frequency = 0.2 + (i * 0.02)
            amplitude = 1000 + (i * 100)
            offset = 1500

            # Generate a value that oscillates
            value = int(offset + amplitude * np.sin(frequency * time_val + phase))
            test_data.append(value)

        # Process the simulated data
        self.process_data(test_data)

    def update_graph_visibility(self):
        """Update which channels are visible on the graph"""
        for i in range(self.channel_count):
            visible = self.channel_checkboxes[i].isChecked()
            if visible and self.graph_time_data:
                self.plot_lines[i].setData(self.graph_time_data, self.graph_data[i])
            else:
                self.plot_lines[i].setData([], [])

    def set_all_channels(self, state):
        """Set all channel checkboxes to the given state"""
        for cb in self.channel_checkboxes:
            cb.setChecked(state)

    def get_channel_color(self, index, qt_color=False, css=False):
        """Get a color for a channel based on its index"""
        # Define some distinct colors
        colors = [
            (0, 0, 255),  # Blue
            (255, 0, 0),  # Red
            (0, 200, 0),  # Green
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (255, 69, 0),  # Red-Orange
            (0, 0, 128),  # Navy
            (255, 255, 0),  # Yellow
            (0, 128, 0)  # Dark Green
        ]

        r, g, b = colors[index % len(colors)]
        if qt_color:
            return QColor(r, g, b)
        elif css:
            return f"rgb({r}, {g}, {b})"
        else:
            # Create a QBrush for table cell background
            color = QColor(r, g, b, 100)  # Semi-transparent
            return color

    def closeEvent(self, event):
        """Clean up when the application is closing"""
        # Stop the serial worker thread
        if self.serial_worker.is_running:
            self.serial_worker.disconnect_serial()
            self.serial_worker.wait()

        # Close log file if open
        if self.log_file:
            self.log_file.close()

        # Stop performance mode if active
        if hasattr(self, 'perf_mode_timer') and self.perf_mode_timer.isActive():
            self.perf_mode_timer.stop()

        # Accept the close event
        event.accept()


# Additional utility function to parse your specific data format
def parse_hex_data(hex_string):
    """
    Parse a hex string format like "0B EB 0B E1 0B E2 0B E4 0B E1 0B E0 0B DE 0B E6 01 4A 01 5A 31"
    Returns a list of values for each channel
    """
    # Remove any whitespace and split by spaces
    hex_values = hex_string.strip().split()

    # Convert hex strings to integers
    values = []
    for i in range(0, len(hex_values) - 1, 2):  # Skipping the last byte which might be a checksum
        if i + 1 < len(hex_values):
            high_byte = int(hex_values[i], 16)
            low_byte = int(hex_values[i + 1], 16)
            value = (high_byte << 8) | low_byte
            values.append(value)

    return values


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AerationDetectionSystem()
    window.show()

    # Test with sample data provided by the user
    sample_data = "0B EB 0B E1 0B E2 0B E4 0B E1 0B E0 0B DE 0B E6 01 4A 01 5A 31"
    parsed_data = parse_hex_data(sample_data)
    print(f"Sample data parsed: {parsed_data}")

    sys.exit(app.exec_())