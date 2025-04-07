import sys
import time
import serial # Using pyserial
import serial.tools.list_ports
import numpy as np
import struct # Added for unpacking binary data
import collections # <--- ADD THIS LINE
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QLineEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QGroupBox, QGridLayout, QCheckBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
# Import QIntValidator for input validation
from PyQt5.QtGui import QColor, QIntValidator # <-- Added QIntValidator
import pyqtgraph as pg

# --- Constants ---
NUM_CHANNELS = 10
MESSAGE_LENGTH = 21 # 10 channels * 2 bytes + 1 frame tail byte
PLOT_MAX_POINTS = 100 # Renamed for clarity
UI_UPDATE_INTERVAL_MS = 150 # Slightly longer interval for UI updates

class SerialWorker(QThread):
    """
    Worker thread to handle serial communication using pyserial.
    """
    # Signals to communicate with the main GUI thread
    dataReceived = pyqtSignal(list) # Emits list of 10 raw integer values
    connectionStatus = pyqtSignal(bool, str) # Emits connection success/failure and port name/error message
    message = pyqtSignal(str) # Signal for general status messages/errors from worker

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.port = None
        self.baud_rate = None
        self.is_running = False
        self.buffer = bytearray()

    def connect_serial(self, port, baud_rate):
        """Stores connection details. Actual connection happens in run()."""
        self.port = port
        self.baud_rate = baud_rate
        self.is_running = True # Signal the run loop to start attempting connection

    def disconnect_serial(self):
        """Signals the run loop to stop and close the port."""
        self.is_running = False

    def run(self):
        """Main loop for the serial worker thread."""
        self.message.emit(f"Worker thread started for {self.port} @ {self.baud_rate} baud.")

        while self.is_running:
            # --- Connection Attempt ---
            if self.serial_port is None or not self.serial_port.is_open:
                try:
                    self.message.emit(f"Attempting to connect to {self.port}...")
                    # Set a timeout for read operations
                    self.serial_port = serial.Serial(self.port, self.baud_rate, timeout=0.1) # Shorter timeout
                    self.buffer.clear() # Clear buffer on successful connection
                    self.connectionStatus.emit(True, self.port) # Signal success
                    self.message.emit(f"Successfully connected to {self.port}.")
                except serial.SerialException as e:
                    # More specific exception handling
                    error_msg = f"Serial connection error on {self.port}: {e}"
                    self.message.emit(error_msg)
                    self.connectionStatus.emit(False, error_msg)
                    self.serial_port = None # Ensure port object is None on failure
                    self.is_running = False # Stop trying if connection fails initially
                    # Optional: could add a retry mechanism here instead of stopping
                    time.sleep(1) # Wait a bit before exiting loop if connection fails
                    continue # Skip reading attempt if connection failed
                except Exception as e:
                    # Catch other potential errors
                    error_msg = f"Unexpected error connecting to {self.port}: {e}"
                    self.message.emit(error_msg)
                    self.connectionStatus.emit(False, error_msg)
                    self.serial_port = None
                    self.is_running = False
                    time.sleep(1)
                    continue

            # --- Data Reading ---
            if self.serial_port and self.serial_port.is_open:
                try:
                    # Check if bytes are available to read
                    if self.serial_port.in_waiting > 0:
                        # Read all available bytes
                        data = self.serial_port.read(self.serial_port.in_waiting)
                        self.buffer.extend(data)

                        # Process complete messages from the buffer
                        while len(self.buffer) >= MESSAGE_LENGTH:
                            message_bytes = self.buffer[:MESSAGE_LENGTH]
                            # Remove the processed message bytes immediately
                            self.buffer = self.buffer[MESSAGE_LENGTH:]

                            data_bytes = message_bytes[:20] # First 20 bytes are data
                            frame_tail = message_bytes[20]  # Last byte is frame tail

                            # --- Optional: Validate Frame Tail ---
                            # expected_tail = 0x31
                            # if frame_tail != expected_tail:
                            #     self.message.emit(f"Warning: Invalid frame tail! Expected {expected_tail:02X}, Got {frame_tail:02X}")
                            #     continue # Discard message

                            # --- Unpack Data using struct ---
                            try:
                                # '>': Big-Endian, '10': Ten values, 'H': Unsigned Short (16-bit)
                                channel_values = list(struct.unpack(f'>{NUM_CHANNELS}H', data_bytes))
                                # Emit the valid data
                                self.dataReceived.emit(channel_values)
                            except struct.error as e:
                                self.message.emit(f"Error unpacking data: {e}. Data: {data_bytes.hex()}")
                                # Data might be corrupted, loop continues with remaining buffer

                except serial.SerialException as e:
                    # Handle potential errors during read (e.g., device disconnected)
                    error_msg = f"Serial read error: {e}"
                    self.message.emit(error_msg)
                    self.connectionStatus.emit(False, error_msg)
                    if self.serial_port and self.serial_port.is_open:
                        self.serial_port.close()
                    self.serial_port = None
                    self.is_running = False # Stop thread on read error
                    time.sleep(0.5)
                except Exception as e:
                    # Catch other unexpected errors during reading/processing
                    error_msg = f"Unexpected error in worker loop: {e}"
                    self.message.emit(error_msg)
                    time.sleep(0.5) # Prevent high CPU usage on continuous errors

            # Small sleep to prevent busy-waiting if no data arrives frequently
            # Timeout in serial.Serial also helps here
            if self.is_running:
                 time.sleep(0.01) # Short sleep even if connected


        # --- Cleanup after loop exits ---
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.message.emit(f"Closed serial port {self.port}.")
        self.serial_port = None
        self.connectionStatus.emit(False, "Disconnected") # Ensure final status is disconnected
        self.message.emit("Worker thread finished.")


class AerationDetectionSystem(QMainWindow):
    """
    Main application window.
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("水流掺气检测系统")
        self.resize(1200, 700)

        # Data storage
        self.channel_count = NUM_CHANNELS
        self.raw_values = [0] * self.channel_count
        self.zero_points = [0.0] * self.channel_count # Use float for calibration points
        self.full_points = [3000.0] * self.channel_count# Use float for calibration points
        self.aeration_rates = [0.0] * self.channel_count

        # Graph data storage (using deques for efficiency)
        self.graph_time_data = collections.deque(maxlen=PLOT_MAX_POINTS)
        self.graph_data = [collections.deque(maxlen=PLOT_MAX_POINTS) for _ in range(self.channel_count)]
        self.graph_start_time = None

        # Serial worker setup
        self.serial_worker = SerialWorker()
        self.serial_worker.dataReceived.connect(self.process_data)
        self.serial_worker.connectionStatus.connect(self.update_connection_status)
        self.serial_worker.message.connect(self.log_worker_message) # Log messages from worker

        # UI Setup
        self.is_demo_mode = False
        self.setup_ui()

        # UI Update Timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(UI_UPDATE_INTERVAL_MS)

        # Logging setup
        self.logging_enabled = False
        self.log_file = None

        # Demo mode timer
        self.perf_mode_timer = QTimer()
        self.perf_mode_timer.timeout.connect(self.generate_test_data)

    def log_worker_message(self, message):
        """Logs messages received from the worker thread."""
        print(f"Worker: {message}")
        # Optionally display important worker messages in status bar too
        # self.statusBar().showMessage(message, 3000) # Show for 3 seconds

    def setup_ui(self):
        """Set up the user interface"""
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # --- Control Panel ---
        control_group = QGroupBox("控制面板")
        control_layout = QGridLayout()

        control_layout.addWidget(QLabel("串口:"), 0, 0)
        self.port_selector = QComboBox()
        self.refresh_ports()
        control_layout.addWidget(self.port_selector, 0, 1)

        control_layout.addWidget(QLabel("波特率:"), 0, 2)
        self.baud_selector = QComboBox()
        self.baud_selector.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baud_selector.setCurrentText("115200")
        control_layout.addWidget(self.baud_selector, 0, 3)

        self.connect_button = QPushButton("连接")
        self.connect_button.clicked.connect(self.toggle_connection)
        control_layout.addWidget(self.connect_button, 0, 4)

        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: red")
        control_layout.addWidget(self.status_indicator, 0, 5)

        self.refresh_ports_button = QPushButton("刷新串口") # Renamed variable
        self.refresh_ports_button.clicked.connect(self.refresh_ports)
        control_layout.addWidget(self.refresh_ports_button, 0, 6)

        # --- Row 2 Controls ---
        self.log_button = QPushButton("开始记录")
        self.log_button.clicked.connect(self.toggle_logging)
        control_layout.addWidget(self.log_button, 1, 0, 1, 2) # Span 2 columns

        self.perf_button = QPushButton("进入演示模式")
        self.perf_button.clicked.connect(self.toggle_performance_mode)
        control_layout.addWidget(self.perf_button, 1, 2, 1, 2) # Span 2 columns

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # --- Calibration Panel ---
        calib_group = QGroupBox("校准设置")
        calib_layout = QGridLayout()

        calib_layout.addWidget(QLabel("通道:"), 0, 0)
        self.channel_selector = QComboBox()
        self.channel_selector.addItems([f"{i+1}" for i in range(self.channel_count)])
        self.channel_selector.currentIndexChanged.connect(self.update_calibration_inputs) # Update inputs on change
        calib_layout.addWidget(self.channel_selector, 0, 1)

        calib_layout.addWidget(QLabel("零点值:"), 0, 2)
        self.zero_input = QLineEdit("0")
        self.zero_input.setValidator(QIntValidator(0, 65535)) # Validate input as integer
        calib_layout.addWidget(self.zero_input, 0, 3)

        calib_layout.addWidget(QLabel("满点值:"), 0, 4)
        self.full_input = QLineEdit("3000")
        self.full_input.setValidator(QIntValidator(0, 65535)) # Validate input as integer
        calib_layout.addWidget(self.full_input, 0, 5)

        self.set_calib_button = QPushButton("设置校准")
        self.set_calib_button.clicked.connect(self.set_calibration)
        calib_layout.addWidget(self.set_calib_button, 0, 6)

        self.get_zero_button = QPushButton("获取当前值为零点")
        self.get_zero_button.clicked.connect(self.get_current_as_zero)
        self.get_zero_button.setEnabled(False) # Disabled initially
        calib_layout.addWidget(self.get_zero_button, 1, 2, 1, 2)

        self.get_full_button = QPushButton("获取当前值为满点")
        self.get_full_button.clicked.connect(self.get_current_as_full)
        self.get_full_button.setEnabled(False) # Disabled initially
        calib_layout.addWidget(self.get_full_button, 1, 4, 1, 2)

        calib_group.setLayout(calib_layout)
        main_layout.addWidget(calib_group)

        # --- Data Display and Graph Section ---
        data_graph_layout = QHBoxLayout()

        # Data Table
        data_display_group = QGroupBox("数据显示")
        data_layout = QVBoxLayout()
        self.data_table = QTableWidget(self.channel_count, 5)
        self.data_table.setHorizontalHeaderLabels(["通道", "原始值", "零点", "满点", "掺气率(%)"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers) # Read-only
        self.data_table.setAlternatingRowColors(True)

        # Initialize table
        for i in range(self.channel_count):
            channel_item = QTableWidgetItem(f"通道 {i+1}")
            color = self.get_channel_color(i, qt_color=True)
            channel_item.setForeground(color if color.lightness() < 128 else QColor('black')) # Text color contrast
            channel_item.setBackground(self.get_channel_color(i)) # Background color
            channel_item.setTextAlignment(Qt.AlignCenter)
            self.data_table.setItem(i, 0, channel_item)
            # Set initial values (text alignment centered)
            for col in range(1, 5):
                 item = QTableWidgetItem()
                 item.setTextAlignment(Qt.AlignCenter)
                 self.data_table.setItem(i, col, item)
            self.data_table.item(i, 1).setText("0")
            self.data_table.item(i, 2).setText(f"{self.zero_points[i]:.0f}") # Display initial as int/float
            self.data_table.item(i, 3).setText(f"{self.full_points[i]:.0f}")
            self.data_table.item(i, 4).setText("0.00")

        data_layout.addWidget(self.data_table)
        data_display_group.setLayout(data_layout)
        # Give table less horizontal space compared to graph
        data_graph_layout.addWidget(data_display_group, 1) # Stretch factor 1

        # Graph Section
        graph_group = QGroupBox("掺气率实时曲线")
        graph_layout = QVBoxLayout()

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setLabel('left', '掺气率 (%)')
        self.graph_widget.setLabel('bottom', '时间 (秒)')
        self.graph_widget.showGrid(x=True, y=True, alpha=0.3)
        self.graph_widget.setYRange(0, 100) # Set fixed Y range
        self.graph_legend = self.graph_widget.addLegend()

        self.plot_lines = []
        for i in range(self.channel_count):
            pen = pg.mkPen(color=self.get_channel_color(i, qt_color=True), width=2)
            line = self.graph_widget.plot([], [], pen=pen, name=f"通道 {i+1}")
            self.plot_lines.append(line)

        graph_layout.addWidget(self.graph_widget)

        # Channel visibility controls
        channel_controls_layout = QGridLayout() # Use grid for checkboxes too
        channel_controls_layout.addWidget(QLabel("显示通道:"), 0, 0)
        self.channel_checkboxes = []
        for i in range(self.channel_count):
             cb = QCheckBox(f"{i+1}") # Just number is cleaner
             cb.setChecked(True)
             cb.stateChanged.connect(self.update_graph_visibility)
             # Style checkbox text color to match line color
             cb.setStyleSheet(f"QCheckBox {{ color: {self.get_channel_color(i, css=True)}; }}")
             self.channel_checkboxes.append(cb)
             # Arrange checkboxes in grid (e.g., 2 rows)
             row, col = divmod(i, 5)
             channel_controls_layout.addWidget(cb, row, col + 1) # Start from column 1

        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(lambda: self.set_all_channels(True))
        channel_controls_layout.addWidget(select_all_btn, 2, 1, 1, 2) # Span button

        select_none_btn = QPushButton("取消")
        select_none_btn.clicked.connect(lambda: self.set_all_channels(False))
        channel_controls_layout.addWidget(select_none_btn, 2, 3, 1, 2) # Span button

        graph_layout.addLayout(channel_controls_layout)
        graph_group.setLayout(graph_layout)
        # Give graph more horizontal space
        data_graph_layout.addWidget(graph_group, 3) # Stretch factor 3

        main_layout.addLayout(data_graph_layout)

        # Status bar setup
        self.status_bar_permanent_widget = QLabel("未连接")
        self.statusBar().addPermanentWidget(self.status_bar_permanent_widget)
        self.statusBar().showMessage("就绪") # Initial message

        self.setCentralWidget(central_widget)
        self.update_calibration_inputs() # Update inputs for initially selected channel

    def refresh_ports(self):
        """Refresh the list of available serial ports"""
        self.port_selector.clear()
        ports = serial.tools.list_ports.comports()
        port_devices = [port.device for port in ports]
        if port_devices:
            self.port_selector.addItems(port_devices)
            self.statusBar().showMessage(f"发现 {len(port_devices)} 个串口设备", 3000)
        else:
            self.statusBar().showMessage("未检测到串口", 3000)

    def toggle_connection(self):
        """Connect or disconnect from the serial port via the worker thread"""
        if self.is_demo_mode:
            self.statusBar().showMessage("请先退出演示模式", 3000)
            return

        if not self.serial_worker.is_running:
            port = self.port_selector.currentText()
            baud_rate_str = self.baud_selector.currentText()
            if not port:
                self.statusBar().showMessage("错误: 未选择串口", 3000)
                return
            if not baud_rate_str:
                self.statusBar().showMessage("错误: 未选择波特率", 3000) # Should not happen
                return

            baud_rate = int(baud_rate_str)
            self.clear_data_and_plot() # Clear old data before connecting
            # Disable controls during connection attempt
            self.port_selector.setEnabled(False)
            self.baud_selector.setEnabled(False)
            self.refresh_ports_button.setEnabled(False)
            self.connect_button.setText("连接中...")
            self.connect_button.setEnabled(False)
            self.perf_button.setEnabled(False) # Cannot enter demo while connecting

            self.serial_worker.connect_serial(port, baud_rate)
            self.serial_worker.start() # Start the thread execution
        else:
            # Stop the worker thread
            self.connect_button.setText("断开中...")
            self.connect_button.setEnabled(False)
            self.serial_worker.disconnect_serial()
            # Worker signals disconnection status update

    def update_connection_status(self, connected, port_or_error):
        """Update the UI based on the connection status signal from the worker"""
        if connected:
            self.status_indicator.setStyleSheet("color: lime") # Use lime for better visibility
            self.connect_button.setText("断开连接")
            self.status_bar_permanent_widget.setText(f"已连接: {port_or_error}")
            self.statusBar().showMessage("连接成功", 2000)
            # Enable/Disable controls
            self.connect_button.setEnabled(True)
            self.get_zero_button.setEnabled(True)
            self.get_full_button.setEnabled(True)
            self.perf_button.setEnabled(False) # Cannot enter demo while connected
            self.port_selector.setEnabled(False)
            self.baud_selector.setEnabled(False)
            self.refresh_ports_button.setEnabled(False)

        else:
            self.status_indicator.setStyleSheet("color: red")
            self.connect_button.setText("连接")
            self.status_bar_permanent_widget.setText("未连接")
            if "Disconnected" not in port_or_error : # Only show errors, not standard disconnect message
                self.statusBar().showMessage(f"连接失败: {port_or_error}", 5000)
            else:
                self.statusBar().showMessage("已断开连接", 2000)
            # Enable/Disable controls
            self.connect_button.setEnabled(True)
            self.get_zero_button.setEnabled(False)
            self.get_full_button.setEnabled(False)
            self.perf_button.setEnabled(True) # Can enter demo when disconnected
            self.port_selector.setEnabled(True)
            self.baud_selector.setEnabled(True)
            self.refresh_ports_button.setEnabled(True)

            # If thread stopped due to error, ensure it can be restarted
            if not self.serial_worker.isFinished():
                 self.serial_worker.wait() # Wait if it hasn't finished cleanup

    def process_data(self, data):
        """Process the received list of 10 raw integer values"""
        if len(data) == self.channel_count:
            self.raw_values = data

            # --- Calculate Aeration Rates ---
            for i in range(self.channel_count):
                zero_val = self.zero_points[i]
                full_val = self.full_points[i]
                full_range = full_val - zero_val
                try:
                    if abs(full_range) < 1e-9: # Check for zero range
                        self.aeration_rates[i] = 0.0
                    else:
                        # Calculate percentage, ensure float division
                        normalized_value = (float(data[i]) - zero_val) / full_range
                        # Clamp result between 0 and 100
                        self.aeration_rates[i] = max(0.0, min(100.0, normalized_value * 100.0))
                except Exception as e:
                    print(f"Error calculating rate for channel {i+1}: {e}")
                    self.aeration_rates[i] = 0.0 # Default to 0 on error

            # --- Logging ---
            if self.logging_enabled and self.log_file:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Millisecond precision
                    raw_str = ",".join(map(str, self.raw_values))
                    rate_str = ",".join([f"{rate:.2f}" for rate in self.aeration_rates])
                    log_line = f"{timestamp},{raw_str},{rate_str}\n"
                    self.log_file.write(log_line)
                    # Consider flushing less frequently if performance is an issue
                    # self.log_file.flush()
                except Exception as e:
                    self.statusBar().showMessage(f"写入日志错误: {e}", 3000)
                    self.toggle_logging() # Stop logging on error

            # --- Add data to graph buffers ---
            current_time = time.time()
            if self.graph_start_time is None:
                self.graph_start_time = current_time

            elapsed_time = current_time - self.graph_start_time
            self.graph_time_data.append(elapsed_time)

            for i in range(self.channel_count):
                self.graph_data[i].append(self.aeration_rates[i])
            # Note: Deques handle maxlen automatically, no need for pop(0)


    def update_ui(self):
        """Update the UI table and graph periodically"""
        # --- Update Table ---
        for i in range(self.channel_count):
            # Check if table items exist before setting text (more robust)
            raw_item = self.data_table.item(i, 1)
            zero_item = self.data_table.item(i, 2)
            full_item = self.data_table.item(i, 3)
            rate_item = self.data_table.item(i, 4)

            if raw_item: raw_item.setText(str(self.raw_values[i]))
            if zero_item: zero_item.setText(f"{self.zero_points[i]:.0f}") # Show calibration as int/float
            if full_item: full_item.setText(f"{self.full_points[i]:.0f}")
            if rate_item: rate_item.setText(f"{self.aeration_rates[i]:.2f}")

        # --- Update Graph ---
        # Convert deques to lists for plotting
        time_data_list = list(self.graph_time_data)
        for i in range(self.channel_count):
            if self.channel_checkboxes[i].isChecked() and time_data_list:
                graph_data_list = list(self.graph_data[i])
                 # Ensure data length matches time length (might mismatch slightly if deque maxlen reached between updates)
                if len(graph_data_list) == len(time_data_list):
                    self.plot_lines[i].setData(time_data_list, graph_data_list)
                # else: handle mismatch if necessary, e.g. by padding or slicing
            # No need to explicitly hide, setData with empty lists if checkbox is off (done in update_graph_visibility)

        # --- Update Status Bar Time ---
        # Only update time if not showing a temporary message
        # This simple check might still overwrite messages quickly.
        # A more robust way involves another timer or checking message age.
        # if not self.statusBar().currentMessage().startswith(("错误", "无法", "开始", "停止", "通道", "发现", "已", "连接中", "断开中")):
        #      self.statusBar().showMessage(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Let's simplify: temporary messages use timeout, permanent uses permanent widget
        pass # Remove time update from status bar, use permanent widget for connection status


    def update_calibration_inputs(self):
        """Update calibration line edits when channel selector changes"""
        try:
            channel_idx = self.channel_selector.currentIndex()
            if 0 <= channel_idx < self.channel_count:
                self.zero_input.setText(f"{self.zero_points[channel_idx]:.0f}")
                self.full_input.setText(f"{self.full_points[channel_idx]:.0f}")
        except Exception as e:
            print(f"Error updating calibration inputs: {e}")


    def set_calibration(self):
        """Set calibration values for the selected channel"""
        try:
            channel_idx = self.channel_selector.currentIndex()
            # Read from LineEdits, validator ensures they are ints
            zero_value = float(self.zero_input.text()) # Store as float
            full_value = float(self.full_input.text()) # Store as float

            if abs(full_value - zero_value) < 1e-9:
                self.statusBar().showMessage("错误: 满点值必须大于零点值", 3000)
                return

            self.zero_points[channel_idx] = zero_value
            self.full_points[channel_idx] = full_value

            # Update table immediately
            self.data_table.item(channel_idx, 2).setText(f"{zero_value:.0f}")
            self.data_table.item(channel_idx, 3).setText(f"{full_value:.0f}")

            # Recalculate and update rate for this channel immediately
            self.aeration_rates[channel_idx] = self.calculate_aeration(self.raw_values[channel_idx], zero_value, full_value)
            self.data_table.item(channel_idx, 4).setText(f"{self.aeration_rates[channel_idx]:.2f}")

            self.statusBar().showMessage(f"通道 {channel_idx+1} 校准设置已应用", 2000)
        except ValueError:
            # Should not happen with QIntValidator, but keep as fallback
            self.statusBar().showMessage("错误: 请输入有效的整数", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"设置校准时出错: {e}", 3000)


    def get_current_as_zero(self):
        """Set current raw value as zero point for selected channel"""
        if not self.serial_worker.is_running and not self.is_demo_mode:
             self.statusBar().showMessage("请先连接设备或进入演示模式", 3000)
             return
        try:
            channel_idx = self.channel_selector.currentIndex()
            current_value = self.raw_values[channel_idx]
            self.zero_input.setText(str(current_value)) # Set input field
            self.set_calibration() # Apply the change
            self.statusBar().showMessage(f"通道 {channel_idx+1} 零点值已设置为当前值 {current_value}", 2000)
        except Exception as e:
            self.statusBar().showMessage(f"设置零点时出错: {e}", 3000)

    def get_current_as_full(self):
        """Set current raw value as full point for selected channel"""
        if not self.serial_worker.is_running and not self.is_demo_mode:
             self.statusBar().showMessage("请先连接设备或进入演示模式", 3000)
             return
        try:
            channel_idx = self.channel_selector.currentIndex()
            current_value = self.raw_values[channel_idx]
            self.full_input.setText(str(current_value)) # Set input field
            self.set_calibration() # Apply the change
            self.statusBar().showMessage(f"通道 {channel_idx+1} 满点值已设置为当前值 {current_value}", 2000)
        except Exception as e:
            self.statusBar().showMessage(f"设置满点时出错: {e}", 3000)


    def toggle_logging(self):
        """Toggle data logging to a CSV file on/off"""
        if not self.logging_enabled:
            try:
                log_filename = f"aeration_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                # Use 'utf-8-sig' encoding for better compatibility with Excel (BOM)
                self.log_file = open(log_filename, 'w', newline='', encoding='utf-8-sig')

                # Write header row
                header = ["Timestamp"]
                header.extend([f"Raw_{i+1}" for i in range(self.channel_count)])
                header.extend([f"Rate_{i+1}" for i in range(self.channel_count)])
                self.log_file.write(",".join(header) + "\n")

                self.logging_enabled = True
                self.log_button.setText("停止记录")
                self.statusBar().showMessage(f"开始记录数据到 {log_filename}", 3000)
            except Exception as e:
                self.statusBar().showMessage(f"无法创建日志文件: {e}", 5000)
                if self.log_file:
                     self.log_file.close()
                self.log_file = None
        else:
            self.logging_enabled = False # Set flag first
            if self.log_file:
                try:
                    self.log_file.close()
                    self.statusBar().showMessage("数据记录已停止", 2000)
                except Exception as e:
                    self.statusBar().showMessage(f"关闭日志文件时出错: {e}", 3000)
                finally:
                    self.log_file = None
            self.log_button.setText("开始记录")

    def toggle_performance_mode(self):
        """Toggle performance/demo mode (generate simulated data)"""
        # --- Decision Point: Keep or Remove Demo Mode ---
        # If you want to remove it based on earlier requests, delete this
        # entire method and the corresponding button in setup_ui.
        # ---

        if self.perf_mode_timer.isActive():
            # Stop performance mode
            self.perf_mode_timer.stop()
            self.perf_button.setText("进入演示模式")
            self.status_indicator.setStyleSheet("color: red") # Show disconnected status
            self.status_bar_permanent_widget.setText("未连接")
            self.statusBar().showMessage("演示模式已停止", 2000)
            self.is_demo_mode = False
            self.clear_data_and_plot() # Clear demo data
            # Re-enable connection controls
            self.connect_button.setEnabled(True)
            self.port_selector.setEnabled(True)
            self.baud_selector.setEnabled(True)
            self.refresh_ports_button.setEnabled(True)
            self.get_zero_button.setEnabled(False) # Disable cal buttons
            self.get_full_button.setEnabled(False)

        elif not self.serial_worker.is_running:
            # Start generating simulated data
            self.clear_data_and_plot() # Clear previous data
            self.is_demo_mode = True
            self.perf_mode_timer.start(100) # Generate data every 100ms
            self.perf_button.setText("退出演示模式")
            self.status_indicator.setStyleSheet("color: blue") # Blue for demo mode
            self.status_bar_permanent_widget.setText("演示模式")
            self.statusBar().showMessage("演示模式已启动", 2000)
            # Disable connection controls, enable calibration buttons
            self.connect_button.setEnabled(False)
            self.port_selector.setEnabled(False)
            self.baud_selector.setEnabled(False)
            self.refresh_ports_button.setEnabled(False)
            self.get_zero_button.setEnabled(True)
            self.get_full_button.setEnabled(True)
        else:
            self.statusBar().showMessage("请先断开串口连接以启用演示模式", 3000)

    def generate_test_data(self):
        """Generate simulated sine wave data for demo mode"""
        time_val = time.time()
        test_data = []
        for i in range(self.channel_count):
            phase = i * 0.5
            frequency = 0.2 + (i * 0.02)
            # Ensure amplitude calculation results in variation within typical range
            amplitude = (self.full_points[i] - self.zero_points[i]) * (0.4 + i * 0.05)
            offset = self.zero_points[i] + (self.full_points[i] - self.zero_points[i]) * 0.5

            value = int(offset + amplitude * np.sin(frequency * time_val + phase))
            # Ensure generated value stays within reasonable bounds (e.g., 0-65535 if using 'H')
            value = max(0, min(value, 65535))
            test_data.append(value)

        # Process the simulated data just like real data
        self.process_data(test_data)

    def update_graph_visibility(self):
        """Update which plot lines are visible based on checkboxes"""
        time_data_list = list(self.graph_time_data)
        for i in range(self.channel_count):
            if self.channel_checkboxes[i].isChecked() and time_data_list:
                 graph_data_list = list(self.graph_data[i])
                 if len(graph_data_list) == len(time_data_list): # Check length consistency
                    self.plot_lines[i].setData(time_data_list, graph_data_list)
                 self.plot_lines[i].show() # Ensure visible
            else:
                self.plot_lines[i].clear() # Clear data if hidden
                self.plot_lines[i].hide() # Ensure hidden


    def set_all_channels(self, state):
        """Set all channel checkboxes to the given state (checked/unchecked)"""
        for cb in self.channel_checkboxes:
            # Block signals during programmatic change to avoid multiple updates
            cb.blockSignals(True)
            cb.setChecked(state)
            cb.blockSignals(False)
        # Trigger a single update after all checkboxes are set
        self.update_graph_visibility()

    def clear_data_and_plot(self):
         """Clears current raw values, rates, and plot data."""
         self.raw_values = [0] * self.channel_count
         self.aeration_rates = [0.0] * self.channel_count
         self.graph_time_data.clear()
         self.graph_start_time = None
         for i in range(self.channel_count):
              self.graph_data[i].clear()
         # Immediately update UI to reflect cleared state
         self.update_ui()
         self.update_graph_visibility()


    def get_channel_color(self, index, qt_color=False, css=False):
        """Get a color for a channel based on its index. Returns QColor or CSS string."""
        # Using pyqtgraph's standard color generation for consistency
        color = pg.intColor(index, hues=NUM_CHANNELS * 1.3, values=1, maxValue=255, minValue=100, alpha=255)

        if qt_color:
            return color # Returns QColor object
        elif css:
            return f"rgb({color.red()}, {color.green()}, {color.blue()})"
        else:
            # Return a slightly transparent QColor for background
            bg_color = QColor(color)
            bg_color.setAlpha(70) # Adjust transparency
            return bg_color

    def closeEvent(self, event):
        """Actions to perform when the application window is closed."""
        print("Closing application...")
        # Stop the update timer
        self.update_timer.stop()
        # Stop demo mode if active
        if self.perf_mode_timer.isActive():
            self.perf_mode_timer.stop()
        # Stop the serial worker thread gracefully
        if self.serial_worker.is_running or self.serial_worker.isRunning():
            print("Stopping serial worker...")
            self.serial_worker.disconnect_serial()
            if not self.serial_worker.wait(2000): # Wait up to 2 seconds for thread to finish
                 print("Warning: Serial worker did not terminate gracefully.")
                 self.serial_worker.terminate() # Force terminate if needed (use with caution)

        # Close log file if open
        if self.logging_enabled and self.log_file:
            print("Closing log file...")
            try:
                self.log_file.close()
            except Exception as e:
                print(f"Error closing log file: {e}")

        print("Exiting.")
        event.accept() # Allow the window to close


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply a modern style if desired
    app.setStyle("Fusion")
    window = AerationDetectionSystem()
    window.show()
    sys.exit(app.exec_())