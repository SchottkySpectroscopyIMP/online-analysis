#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys, ast, os, platform
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QStatusBar, 
                            QPushButton, QMessageBox, QDoubleSpinBox, QLineEdit, QFileDialog,
                             QHBoxLayout, QSpinBox, QLabel, QGroupBox, QGridLayout, QCompleter, QRadioButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QStringListModel
from PyQt5.QtGui import QFont
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit

class Worker(QObject):
    # 定义一个信号用来将计算结果传回主线程
    result_ready = pyqtSignal(object, object)

    def __init__(self, df_copy, spin_vals, list_vals):
        super().__init__()
        self.df = df_copy
        self.spin_vals = spin_vals
        self.initial_info = list_vals

    def run_GMM(self):
        labels_name = list(self.initial_info.keys())
        labels_harmonic = [val[1] for val in self.initial_info.values()]
        x_init_list = [val[0] for val in self.initial_info.values()]
        x0, x1, y0, y1 = self.spin_vals
        mask = (self.df['peak_pos']>x0) & (self.df['peak_pos']<x1) & (self.df['height_ion']>y0) & (self.df['height_ion']<y1)
        subset = self.df.loc[mask, ['peak_pos', 'height_ion']]
        filtered_data = subset.values

        if len(filtered_data) == 0:
            self.result_ready.emit(self.df['ion'].values, self.df['harmonic'].values)
            return

        # 准备初始值
        y_init = np.mean(filtered_data[:,1])
        inital_means = np.array([[x, y_init] for x in x_init_list])

        gmm = GaussianMixture(n_components=len(x_init_list), means_init=inital_means, random_state=42)
        gmm.fit(filtered_data)

        # 寻找与每个拟合中心最近的初始 x 索引，计算拟合中心与初始中心之间的映射关系
        fitted_means_x = gmm.means_[:,0]
        mapping_labels, mapping_harmonics = {}, {}
        for i in range(gmm.n_components):
            closest_idx = np.argmin(np.abs(x_init_list-fitted_means_x[i]))
            mapping_labels[i] = labels_name[closest_idx]
            mapping_harmonics[i] = labels_harmonic[closest_idx]

        _labels = gmm.predict(filtered_data)
        self.df.loc[mask, 'ion'] = [mapping_labels[num] for num in _labels]
        self.df.loc[mask, 'harmonic'] = [mapping_harmonics[num] for num in _labels]

        self.result_ready.emit(self.df['ion'].values, self.df['harmonic'].values)


# 双 X 轴 sigma f/f 绘图与拟合控制窗口（精简为单一散点呈现）
class SigmaPlotWindow(QMainWindow):
    def __init__(self, summary_df):
        super().__init__()
        self.setWindowTitle("σf/f vs m/q & γ")
        self.resize(1000, 600)
        self.summary_df = summary_df
        self.apply_global_font()

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.coord_label = QLabel("m/q: 0.0000, γ: 0.0000, σf/f: 0.000000")
        self.status_bar.addPermanentWidget(self.coord_label)

        # 布局构建
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧：绘图控件
        self.plot_widget = pg.PlotWidget()
        main_layout.addWidget(self.plot_widget, stretch=4)

        # 右侧：拟合控制面板
        fit_group = QGroupBox("Fitting Control")
        fit_layout = QGridLayout()

        fit_layout.addWidget(QLabel("γt:"), 0, 0)
        self.spin_gammat = QDoubleSpinBox()
        self.spin_gammat.setRange(0.0001, 1000.0)
        self.spin_gammat.setValue(1.43)
        self.spin_gammat.setDecimals(4)
        self.spin_gammat.setSingleStep(0.01)
        fit_layout.addWidget(self.spin_gammat, 0, 1)

        fit_layout.addWidget(QLabel("ΔΒρ/Βρ (%):"), 1, 0)
        self.spin_dbrp = QDoubleSpinBox()
        self.spin_dbrp.setRange(0.0001, 1000.0)
        self.spin_dbrp.setValue(0.3)
        self.spin_dbrp.setDecimals(4)
        self.spin_dbrp.setSingleStep(0.01)
        fit_layout.addWidget(self.spin_dbrp, 1, 1)

        self.btn_fit = QPushButton("run fitting")
        fit_layout.addWidget(self.btn_fit, 2, 0, 1, 2)

        fit_group.setLayout(fit_layout)
        main_layout.addWidget(fit_group, stretch=1)

        self.btn_fit.clicked.connect(self.run_fitting)

        # 初始化图表
        self.init_plot()

    def apply_global_font(self):
        curr_system = platform.system()

        if curr_system == 'Windows':
            font_family = ['Consolas', 'Microsoft YaHei UI', 'Courier New']
        elif curr_system == 'Darwin':
            font_family = ['Menlo', 'Monaco', 'PingFang SC', 'Heiti SC']
        else:
            font_family = ['Ubuntu Mono', 'Fira Code', 'DejaVu Sans Mono', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']

        font = QFont()
        font.setFamilies(font_family)
        font.setPointSize(10)

        font.setStyleHint(QFont.Monospace)
        QApplication.instance().setFont(font)

    def init_plot(self):
        self.p_main = self.plot_widget.getPlotItem()
        self.p_main.showGrid(x=True, y=True, alpha=0.3)
        self.p_main.setLabel('bottom', 'm/q')
        self.p_main.setLabel('left', 'σf / f')

        # 提取数据真实极值范围，用于建立 m/q 与 gamma 的物理映射
        self.mq_min = self.summary_df['m/q'].min()
        self.mq_max = self.summary_df['m/q'].max()
        self.gamma_min = self.summary_df['gamma'].min()
        self.gamma_max = self.summary_df['gamma'].max()

        if self.mq_max == self.mq_min:
            self.mq_max += 1.0
        if self.gamma_max == self.gamma_min:
            self.gamma_max += 1.0

        # 创建顶部 ViewBox 与顶部 X 轴 (gamma)
        self.vb_top = pg.ViewBox()
        self.p_main.scene().addItem(self.vb_top)
        self.axis_top = pg.AxisItem('top')
        self.axis_top.setLabel('γ')
        self.p_main.layout.addItem(self.axis_top, 1, 1)
        self.axis_top.linkToView(self.vb_top)
        self.vb_top.setYLink(self.p_main.vb)

        # 保持几何区域对齐
        def update_views():
            self.vb_top.setGeometry(self.p_main.vb.sceneBoundingRect())

        # 根据主 ViewBox (m/q) 的范围变化，实时映射顶部 ViewBox (gamma) 的真实范围
        def update_top_x_range():
            x0, x1 = self.p_main.vb.viewRange()[0]
            g0 = self.mq_to_gamma(x0)
            g1 = self.mq_to_gamma(x1)
            self.vb_top.setXRange(g0, g1, padding=0)

        self.p_main.vb.sigResized.connect(update_views)
        self.p_main.vb.sigXRangeChanged.connect(update_top_x_range)

        # 鼠标移动显示坐标
        self.p_main.scene().sigMouseMoved.connect(self.on_mouse_moved)

        # 绘制散点数据
        self.plot_data()

    def mq_to_gamma(self, mq):
        """m/q 到 gamma 的线性换算函数"""
        return self.gamma_min + (mq - self.mq_min) * (self.gamma_max - self.gamma_min) / (self.mq_max - self.mq_min)

    def gamma_to_mq(self, gamma):
        """gamma 到 m/q 的线性换算函数"""
        return self.mq_min + (gamma - self.gamma_min) * (self.mq_max - self.mq_min) / (self.gamma_max - self.gamma_min)

    def plot_data(self):
        if self.summary_df.empty:
            return

        mq_vals = self.summary_df['m/q'].values
        y_vals = self.summary_df['rel_sigma'].values
        ions = self.summary_df['ion'].values

        # 仅在主视图 p_main 上绘制一套散点及标签（上轴 gamma 仅作为坐标标尺）
        scatter_bottom = pg.ScatterPlotItem(
            x=mq_vals, y=y_vals, size=12, symbol='o', 
            brush=pg.mkBrush('cyan'), pen=pg.mkPen('w', width=1)
        )
        self.p_main.addItem(scatter_bottom)

        for x, y, ion in zip(mq_vals, y_vals, ions):
            text_item = pg.TextItem(text=str(ion), color='yellow', anchor=(0.5, 1.2))
            text_item.setPos(x, y)
            self.p_main.addItem(text_item)

        # 设置主视口的范围，触发自动映射更新顶部 view
        dmq = self.mq_max - self.mq_min
        self.p_main.setXRange(self.mq_min - 0.1 * dmq, self.mq_max + 0.1 * dmq)

    def on_mouse_moved(self, evt):
        pos = evt
        if self.p_main.vb.sceneBoundingRect().contains(pos):
            pt_bottom = self.p_main.vb.mapSceneToView(pos)
            pt_top = self.vb_top.mapSceneToView(pos)
            self.coord_label.setText(
                f"m/q: {pt_bottom.x():.4f}, γ: {pt_top.x():.4f}, σf/f: {pt_bottom.y():.6e}"
            )

    def run_fitting(self):
        self.spin_gammat.setEnabled(False)
        self.spin_dbrp.setEnabled(False)
        self.btn_fit.setEnabled(False)

        try:
            init_gammat = self.spin_gammat.value()
            init_dbrp = self.spin_dbrp.value()

            x_data = self.summary_df['gamma'].values
            y_data = self.summary_df['rel_sigma'].values

            # 拟合公式: y = |1/x^2 - 1/γt^2| * ΔΒρ/Βρ * 0.01 + const
            def fit_func(x, g_t, d_brp, const):
                return np.abs(1.0 / (x**2) - 1.0 / (g_t**2)) * d_brp * 0.01 + const

            popt, _ = curve_fit(fit_func, x_data, y_data, p0=[init_gammat, init_dbrp, 0.0], maxfev=5000)
            
            fit_g_t, fit_d_brp, fit_const = popt[0], popt[1], popt[2]

            # 覆盖写入输入框
            self.spin_gammat.setValue(float(fit_g_t))
            self.spin_dbrp.setValue(float(fit_d_brp))

            # 清理旧的拟合曲线
            if hasattr(self, 'fit_curve') and self.fit_curve in self.p_main.items:
                self.p_main.removeItem(self.fit_curve)

            # 在主图 p_main 上绘制拟合曲线（将 gamma 坐标转换为 m/q 绘制，确保对齐）
            gamma_min, gamma_max = x_data.min(), x_data.max()
            gamma_fit = np.linspace(gamma_min, gamma_max, 50)
            y_fit = fit_func(gamma_fit, fit_g_t, fit_d_brp, fit_const)
            mq_fit = self.gamma_to_mq(gamma_fit)

            self.fit_curve = pg.PlotCurveItem(mq_fit, y_fit, pen=pg.mkPen('r', width=2, style=Qt.SolidLine))
            self.p_main.addItem(self.fit_curve)

            self.status_bar.showMessage(f"Fitting succeeded: γt = {fit_g_t:.4f}, ΔΒρ/Βρ = {fit_d_brp:.4f}%, const = {fit_const:.6e}", 5000)

        except Exception as e:
            self.status_bar.showMessage(f"Fitting failed: {str(e)}", 5000)

        finally:
            self.spin_gammat.setEnabled(True)
            self.spin_dbrp.setEnabled(True)
            self.btn_fit.setEnabled(True)


class FastLargeDataPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Large Scale Data Visualization")
        self.resize(1400, 900)
        self.roi = None
        self.sigma_win = None
        self.apply_global_font()
        
        self.init_ui()

    def apply_global_font(self):
        curr_system = platform.system()

        if curr_system == 'Windows':
            font_family = ['Consolas', 'Microsoft YaHei UI', 'Courier New']
        elif curr_system == 'Darwin':
            font_family = ['Menlo', 'Monaco', 'PingFang SC', 'Heiti SC']
        else:
            font_family = ['Ubuntu Mono', 'Fira Code', 'DejaVu Sans Mono', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']

        font = QFont()
        font.setFamilies(font_family)
        font.setPointSize(10)

        font.setStyleHint(QFont.Monospace)
        QApplication.instance().setFont(font)


    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win, 0, 0, 3, 1)

        # 1. 主视觉图
        self.p_main = self.win.addPlot(row=0, col=0)
        self.p_main.showGrid(x=True, y=True, alpha=0.3)
        self.p_main.setLabel('left', 'Height Ion')
        
        # 2. 右侧 Y 直方图
        self.p_hist_y = self.win.addPlot(row=0, col=1)
        self.p_hist_y.setFixedWidth(150)
        self.p_hist_y.setYLink(self.p_main)
        self.p_hist_y.getAxis('left').setStyle(showValues=False)

        # 3. 下方 X 直方图
        self.p_hist_x = self.win.addPlot(row=1, col=0)
        self.p_hist_x.setFixedHeight(150)
        self.p_hist_x.setXLink(self.p_main)

        # 4. 右上角文件控制区域
        file_group = QGroupBox("Data file Control")
        file_layout = QGridLayout()
        self.btn_dataFile = QPushButton('Data file')
        file_layout.addWidget(self.btn_dataFile, 0, 0)
        self.linedit_dataFile = QLineEdit()
        self.linedit_dataFile.setReadOnly(True)
        self.linedit_dataFile.setPlaceholderText("Please select .csv file ...")
        file_layout.addWidget(self.linedit_dataFile, 0, 1)
        self.btn_refFile = QPushButton('Ref. file')
        file_layout.addWidget(self.btn_refFile, 1, 0)
        self.linedit_refFile = QLineEdit()
        self.linedit_refFile.setReadOnly(True)
        self.linedit_refFile.setPlaceholderText("Please select .csv File ...")
        file_layout.addWidget(self.linedit_refFile, 1, 1)
        self.btn_fileLoad = QPushButton('Load files')
        file_layout.addWidget(self.btn_fileLoad, 2, 1)
        file_layout.addWidget(QLabel('More ref.:'), 3, 0)
        self.linedit_refIons = QLineEdit()
        file_layout.addWidget(self.linedit_refIons, 3, 1)
        file_layout.addWidget(QLabel("{'195Os75(1)+':[308000756.01, 204, 324.786], ...}"), 4, 1)
        self.btn_addRef = QPushButton('Add ref.')
        file_layout.addWidget(self.btn_addRef, 5,1)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group, 0, 1)

        self.btn_dataFile.clicked.connect(self.select_dataFile)
        self.btn_refFile.clicked.connect(self.select_refFlie)
        self.btn_fileLoad.clicked.connect(self.load_data)
        self.btn_addRef.clicked.connect(self.add_refIons)

        # 5. 右中部直方图控制区域
        control_group = QGroupBox("Bins Control")
        control_layout = QGridLayout()
        control_layout.addWidget(QLabel("X Bins:"), 0, 0)
        self.spin_x_bins = QSpinBox()
        self.spin_x_bins.setRange(0, 500); self.spin_x_bins.setValue(0)
        control_layout.addWidget(self.spin_x_bins, 0, 1)
        control_layout.addWidget(QLabel("Y Bins:"), 1, 0)
        self.spin_y_bins = QSpinBox()
        self.spin_y_bins.setRange(0, 500); self.spin_y_bins.setValue(0)
        control_layout.addWidget(self.spin_y_bins, 1, 1)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group, 1, 1)

        self.spin_x_bins.valueChanged.connect(self.update_histograms)
        self.spin_y_bins.valueChanged.connect(self.update_histograms)
        # 核心：监听范围变化
        self.p_main.sigRangeChanged.connect(self.update_histograms)

        # 6. 右下角GMM控制区域
        gmmcontrol_group = QGroupBox("GMM Control")
        gmmcontrol_layout = QGridLayout()

        # 单选 ROI 框选按钮
        self.radio_select_roi = QRadioButton('Select Region (ROI)')
        gmmcontrol_layout.addWidget(self.radio_select_roi, 0, 0, 1, 2)

        gmmcontrol_layout.addWidget(QLabel("freq start (Hz):"), 1, 0)
        self.spin_x0 = QDoubleSpinBox()
        self.spin_x0.setRange(0, 1e10); self.spin_x0.setValue(308e6); self.spin_x0.setDecimals(3); self.spin_x0.setSingleStep(0.1)
        gmmcontrol_layout.addWidget(self.spin_x0, 1, 1)
        gmmcontrol_layout.addWidget(QLabel("freq end (Hz):"), 2, 0)
        self.spin_x1 = QDoubleSpinBox()
        self.spin_x1.setRange(0, 1e10); self.spin_x1.setValue(308.5e6); self.spin_x1.setDecimals(3); self.spin_x1.setSingleStep(0.1)
        gmmcontrol_layout.addWidget(self.spin_x1, 2, 1)
        gmmcontrol_layout.addWidget(QLabel("height start:"), 3, 0)
        self.spin_y0 = QDoubleSpinBox()
        self.spin_y0.setRange(-100, 100000); self.spin_y0.setValue(0); self.spin_y0.setDecimals(3); self.spin_y0.setSingleStep(0.2)
        gmmcontrol_layout.addWidget(self.spin_y0, 3, 1)
        gmmcontrol_layout.addWidget(QLabel("height end:"), 4, 0)
        self.spin_y1 = QDoubleSpinBox()
        self.spin_y1.setRange(-100, 100000); self.spin_y1.setValue(10); self.spin_y1.setDecimals(3); self.spin_y1.setSingleStep(0.1)
        gmmcontrol_layout.addWidget(self.spin_y1, 4, 1)

        # Append Ion 按钮与搜索框
        self.btn_append_ion = QPushButton('Append Ion')
        gmmcontrol_layout.addWidget(self.btn_append_ion, 5, 0)

        self.linedit_search_ion = QLineEdit()
        self.linedit_search_ion.setPlaceholderText("e.g. 195Os75(1)+, 204")
        self.ion_completer = QCompleter(self)
        self.ion_completer.setFilterMode(Qt.MatchContains)
        self.ion_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.linedit_search_ion.setCompleter(self.ion_completer)
        gmmcontrol_layout.addWidget(self.linedit_search_ion, 5, 1)

        gmmcontrol_layout.addWidget(QLabel("init ion freqs:"), 6, 0)
        self.linedit_ionfreqs = QLineEdit()
        self.linedit_ionfreqs.setText('{}')
        gmmcontrol_layout.addWidget(self.linedit_ionfreqs, 6, 1)
        gmmcontrol_layout.addWidget(QLabel("{'195Os75(1)+':[308000756.01, 204], \n'195Os75(0)+'}:[308000960.24, 204],...}"), 7, 1)
        self.btn_gmm = QPushButton('GMM run')
        self.btn_clear = QPushButton('Clear result')
        gmmcontrol_layout.addWidget(self.btn_gmm, 8, 1)
        gmmcontrol_layout.addWidget(self.btn_clear, 9, 1)
        self.btn_csvFolder = QPushButton("Folder")
        gmmcontrol_layout.addWidget(self.btn_csvFolder, 10, 0)
        self.linedit_csvFolder = QLineEdit()
        self.linedit_csvFolder.setReadOnly(True)
        self.linedit_csvFolder.setPlaceholderText("Please select folder ...")
        gmmcontrol_layout.addWidget(self.linedit_csvFolder, 10, 1)
        gmmcontrol_layout.addWidget(QLabel("Filename:"), 11, 0)
        self.linedit_filename = QLineEdit()
        self.linedit_filename.setText('')
        gmmcontrol_layout.addWidget(self.linedit_filename, 11, 1)
        self.btn_csvSave = QPushButton('Save .csv')
        gmmcontrol_layout.addWidget(self.btn_csvSave, 12, 1)
        gmmcontrol_group.setLayout(gmmcontrol_layout)
        layout.addWidget(gmmcontrol_group, 2, 1)

        self.radio_select_roi.toggled.connect(self.on_roi_toggled)
        self.btn_append_ion.clicked.connect(self.append_searched_ion)
        self.btn_gmm.clicked.connect(self.gmm_start)
        self.btn_clear.clicked.connect(self.result_clear)
        self.btn_csvFolder.clicked.connect(self.select_folder)
        self.btn_csvSave.clicked.connect(self.save_csv)

        layout.setColumnStretch(0,4)
        layout.setColumnStretch(1,1)

        # 6. 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.coord_label = QLabel("Freq: 0.00, Height: 0.0000")
        self.status_bar.addPermanentWidget(self.coord_label)

    def select_dataFile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select .csv data file", './', "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.linedit_dataFile.setText(file_path)

    def select_refFlie(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select .csv ref. file", './', "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.linedit_refFile.setText(file_path)

    def load_data(self):
        self.btn_fileLoad.setEnabled(False)
        self.btn_fileLoad.setText('Loading files ...')
        try:
            self.data_dir, self.data_name = os.path.split(self.linedit_dataFile.text())
            self.df_data = pd.read_csv(self.linedit_dataFile.text())
            self.df_ref = pd.read_csv(self.linedit_refFile.text())
            self.df_data['ion'] = ''
            self.df_data['harmonic'] = np.nan
            
            # 修正：在 columns 上使用 .str.replace
            self.df_ref.columns = self.df_ref.columns.str.replace(r"\s*\(.*?\)", "", regex=True)
            self.label_items = [] 
            self.current_ion_brushes = None  # 重置填充色缓存

            # 更新补全提示列表
            if 'ion' in self.df_ref.columns and 'harmonic' in self.df_ref.columns:
                ref_items = []
                for _, row in self.df_ref.dropna(subset=['ion', 'harmonic']).iterrows():
                    ref_items.append(f"{row['ion']}, {int(row['harmonic'])}")
                self.ion_completer.setModel(QStringListModel(ref_items, self))

            self.plot_reference_lines()
            self.plot_data()
            self.update_histograms()
            self.linedit_filename.setText(self.data_name)
        except Exception as e:
            QMessageBox.critical(
                self,
                "File loading error",
                f"Error: {e}\n\nPlease check and select files again.",
                QMessageBox.Ok
            )
            self.btn_fileLoad.setEnabled(True)
            self.btn_fileLoad.setText('Load files')
            return
        self.btn_fileLoad.setEnabled(True)
        self.btn_fileLoad.setText('Load files')

    def on_roi_toggled(self, checked):
        if checked:
            view_range = self.p_main.viewRange()
            x_min, x_max = view_range[0][0], view_range[0][1]
            y_min, y_max = view_range[1][0], view_range[1][1]
            
            dx = x_max - x_min
            dy = y_max - y_min
            
            roi_x0 = x_min + 0.2 * dx
            roi_w = 0.6 * dx
            roi_y0 = y_min + 0.2 * dy
            roi_h = 0.6 * dy
            
            if self.roi is not None:
                self.p_main.removeItem(self.roi)
                self.roi = None
            
            self.roi = pg.RectROI([roi_x0, roi_y0], [roi_w, roi_h], pen=pg.mkPen('r', width=2))
            self.p_main.addItem(self.roi)
            self.roi.sigRegionChanged.connect(self.update_spins_from_roi)
            self.update_spins_from_roi()
        else:
            if self.roi is not None:
                self.p_main.removeItem(self.roi)
                self.roi = None

    def update_spins_from_roi(self):
        if self.roi is None:
            return
        pos = self.roi.pos()
        size = self.roi.size()

        x0, x1 = pos.x(), pos.x() + size.x()
        y0, y1 = pos.y(), pos.y() + size.y()

        self.spin_x0.setValue(min(x0, x1))
        self.spin_x1.setValue(max(x0, x1))
        self.spin_y0.setValue(min(y0, y1))
        self.spin_y1.setValue(max(y0, y1))

    def append_searched_ion(self):
        input_text = self.linedit_search_ion.text().strip()
        if not input_text:
            return

        if not hasattr(self, 'df_ref') or self.df_ref is None or self.df_ref.empty:
            QMessageBox.warning(self, "Warning", "Please load reference file first!")
            return

        if ',' in input_text:
            parts = [p.strip() for p in input_text.split(',')]
            ion_name = parts[0]
            try:
                harmonic_val = int(parts[1])
            except ValueError:
                harmonic_val = None
        else:
            ion_name = input_text
            harmonic_val = None

        if harmonic_val is not None:
            matched = self.df_ref[(self.df_ref['ion'] == ion_name) & (self.df_ref['harmonic'] == harmonic_val)]
        else:
            matched = self.df_ref[self.df_ref['ion'] == ion_name]

        if matched.empty:
            QMessageBox.warning(self, "Warning", f"Ion '{input_text}' not found in reference file!")
            return

        row = matched.iloc[0]
        freq_hz = round(float(row['peak_loc'] * 1e3), 2)
        h_val = int(row['harmonic'])

        curr_text = self.linedit_ionfreqs.text().strip()
        try:
            curr_dict = ast.literal_eval(curr_text) if curr_text else {}
            if not isinstance(curr_dict, dict):
                curr_dict = {}
        except Exception:
            curr_dict = {}

        curr_dict[ion_name] = [freq_hz, h_val]
        self.linedit_ionfreqs.setText(str(curr_dict))
        self.linedit_search_ion.clear()

    def add_refIons(self):
        self.btn_addRef.setEnabled(False)
        self.btn_addRef.setText('Adding ions ...')
        try:
            list_vals = ast.literal_eval(self.linedit_refIons.text())
            for ion, ion_info in list_vals.items():
                x_val = ion_info[0]
                ion_label = ion + '\nsigma: {:.3f}\nh = {:d}'.format(ion_info[-1], ion_info[1])

                line = pg.InfiniteLine(pos=x_val, angle=90, pen=pg.mkPen('lightgray', width=1.5, style=Qt.DashLine))
                self.p_main.addItem(line)
                text = pg.TextItem(text=ion_label, color='lightgray', anchor=(0, 0))
                self.p_main.addItem(text)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Input format error",
                f"Input dict format error! \n\nError: {e}\n\nPlease check and input again."+"\n\n\nExample: {'195Os75(1)+':[308000756.01, 204, 324.786], ...}",
                QMessageBox.Ok
            )
            self.btn_addRef.setEnabled(True)
            self.btn_addRef.setText('Add ref.')
            return
            
        try:
            self.label_items.append((x_val, text))
        except Exception as e:
            QMessageBox.critical(
                self,
                "Adding error",
                f"Adding error! \n\nNeeded to load files first!",
                QMessageBox.Ok
            )
            self.btn_addRef.setEnabled(True)
            self.btn_addRef.setText('Add ref.')

        self.btn_addRef.setEnabled(True)
        self.btn_addRef.setText('Add ref.')

    def plot_reference_lines(self):
        for _, row in self.df_ref.iterrows():
            x_val = row['peak_loc'] * 1e3
            ion_label = row['ion'] + '\nsigma: {:.3f}\nh = {:d}'.format(row['peak_sig'] * 1e3, int(row['harmonic']))
            
            line = pg.InfiniteLine(pos=x_val, angle=90, pen=pg.mkPen('lightgray', width=1.5, style=Qt.DashLine))
            self.p_main.addItem(line)

            text = pg.TextItem(text=ion_label, color='lightgray', anchor=(0, 0))
            self.p_main.addItem(text)
            self.label_items.append((x_val, text))

        self.dynamic_labels = []

    def plot_data(self):
        times = self.df_data['exist_time'].values
        t_min = times.min() if len(times) > 0 else 0.0
        t_max = times.max() if len(times) > 0 else 2.9
        if t_max == t_min:
            t_max = t_min + 1e-5

        self.pg_cmap = pg.colormap.getFromMatplotlib('viridis')
        self.scatter = pg.ScatterPlotItem(x=self.df_data['peak_pos'], y=self.df_data['height_ion'], size=10, symbol='o')

        df_pair = self.df_data[self.df_data['pair_num'] != 0]
        x0_pair, x1_pair, y0_pair, y1_pair = [], [], [], []
        for i, group in df_pair.groupby('filename'):
            for j in np.unique(group['pair_num'].values):
                sub1 = group[(group['pair_num'] == j) & (group['exist_state'] == 1)]
                sub2 = group[(group['pair_num'] == j) & (group['exist_state'] == 2)]
                if not sub1.empty and not sub2.empty:
                    x0_pair.append(sub1['peak_pos'].values[0])
                    x1_pair.append(sub2['peak_pos'].values[0])
                    y0_pair.append(sub1['height_ion'].values[0])
                    y1_pair.append(sub2['height_ion'].values[0])

        if x0_pair:
            x_pair = np.vstack((x0_pair, x1_pair, np.full_like(x0_pair, np.nan, dtype=np.float64))).T.ravel()[:-1]
            y_pair = np.vstack((y0_pair, y1_pair, np.full_like(y0_pair, np.nan, dtype=np.float64))).T.ravel()[:-1]
            decay_line = pg.PlotCurveItem(x=x_pair, y=y_pair, connect='finite', pen=pg.mkPen(color=(200, 200, 200, 120), width=1))
            self.p_main.addItem(decay_line)

        self.p_main.addItem(self.scatter)

        if not hasattr(self, 'colorbar'):
            try:
                self.colorbar = pg.ColorBarItem(values=(t_min, t_max), colorMap=self.pg_cmap, label='Exist Time (s)')
            except TypeError:
                self.colorbar = pg.ColorBarItem(values=(t_min, t_max), cmap=self.pg_cmap, label='Exist Time (s)')
            
            if hasattr(self.colorbar, 'setPlot'):
                self.colorbar.setPlot(self.p_main)
            else:
                try:
                    self.colorbar.setImageItem(None, insertIn=self.p_main)
                except TypeError:
                    self.win.addItem(self.colorbar, row=0, col=2)
            
            self.colorbar.sigLevelsChanged.connect(self.on_colorbar_levels_changed)
        else:
            self.colorbar.setLevels((t_min, t_max))

        self.on_colorbar_levels_changed()

        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='white', style=Qt.DotLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color='white', style=Qt.DotLine))
        self.p_main.addItem(self.v_line, ignoreBounds=True)
        self.p_main.addItem(self.h_line, ignoreBounds=True)
        self.p_main.scene().sigMouseMoved.connect(self.on_mouse_moved)

    def on_colorbar_levels_changed(self):
        if not hasattr(self, 'df_data') or self.df_data is None or self.df_data.empty:
            return

        c_min, c_max = self.colorbar.levels()
        if c_max == c_min:
            c_max = c_min + 1e-5

        times = self.df_data['exist_time'].values
        norm_times = np.clip((times - c_min) / (c_max - c_min), 0.0, 1.0)

        qcolors = self.pg_cmap.map(norm_times, mode='qcolor')
        pen_list = [pg.mkPen(color=c, width=1.5) for c in qcolors]

        brush_list = getattr(self, 'current_ion_brushes', None)

        self.scatter.setData(
            x=self.df_data['peak_pos'],
            y=self.df_data['height_ion'],
            size=10,
            pen=pen_list,
            brush=brush_list,
            symbol='o'
        )

    def on_mouse_moved(self, evt):
        pos = evt
        view_box = self.p_main.vb
        if view_box.sceneBoundingRect().contains(pos):
            mouse_point = view_box.mapSceneToView(pos)
            x_val = mouse_point.x()
            y_val = mouse_point.y()
            self.coord_label.setText(f"Freq: {x_val:.2f}, Height: {y_val:.4f}")
            self.v_line.setPos(x_val)
            self.h_line.setPos(y_val)

    def gmm_start(self):
        if self.spin_x0.value() >= self.spin_x1.value() or self.spin_y0.value() >= self.spin_y1.value():
            return

        self.btn_gmm.setEnabled(False)
        self.btn_gmm.setText('GMM running ...')

        spin_vals = [self.spin_x0.value(), self.spin_x1.value(), self.spin_y0.value(), self.spin_y1.value()]
        try:
            list_vals = ast.literal_eval(self.linedit_ionfreqs.text())
        except Exception as e:
            QMessageBox.critical(
                self,
                "Input format error",
                f"Input dict format error! \n\nError: {e}\n\nPlease check and input again."+"\n\n\nExample: {'195Os75(1)+':[308000756.01,204],'195Os75(0)+':[308000960.24,204]}",
                QMessageBox.Ok
            )
            self.btn_gmm.setEnabled(True)
            self.btn_gmm.setText('GMM run')
            return

        df_copy = self.df_data.copy()
        self.gmm_thread = QThread()
        self.gmm_worker = Worker(df_copy, spin_vals, list_vals)
        self.gmm_worker.moveToThread(self.gmm_thread)
        self.gmm_thread.started.connect(self.gmm_worker.run_GMM)
        self.gmm_worker.result_ready.connect(self.gmm_end)

        self.gmm_worker.result_ready.connect(self.gmm_thread.quit)
        self.gmm_thread.finished.connect(self.gmm_thread.deleteLater)
        self.gmm_worker.deleteLater()

        self.gmm_thread.start()

    def gmm_end(self, possible_ion_range, possible_harmonic_range):
        self.df_data['ion'] = possible_ion_range
        self.df_data['harmonic'] = possible_harmonic_range
        
        real_unique_ions = [ion for ion in self.df_data['ion'].unique() if ion != '']
        num_colors = len(real_unique_ions)
        
        if num_colors > 0:
            colors = [pg.intColor(i, num_colors) for i in range(num_colors)]
            ion_color_map = dict(zip(real_unique_ions, colors))
        else:
            ion_color_map = {}

        self.current_ion_brushes = [
            pg.mkBrush(ion_color_map[ion]) if ion != '' else None 
            for ion in self.df_data['ion']
        ]

        self.on_colorbar_levels_changed()

        if not hasattr(self, 'dynamic_labels'):
            self.dynamic_labels = []
        for old_text_item in self.dynamic_labels:
            self.p_main.removeItem(old_text_item)
        self.dynamic_labels.clear()

        for ion_name in real_unique_ions:
            df_sub = self.df_data[self.df_data['ion'] == ion_name]
            if df_sub.empty:
                continue
            unique_harmonics = df_sub['harmonic'].unique()
            for h in unique_harmonics:
                _df_sub = df_sub[df_sub['harmonic'] == h]
                x_center = _df_sub['peak_pos'].mean()
                y_position = _df_sub['height_ion'].min()
                
                text_item = pg.TextItem(text=ion_name, color=ion_color_map[ion_name], anchor=(0.5, 0))
                text_item.setPos(x_center, y_position)
                self.p_main.addItem(text_item)
                self.dynamic_labels.append(text_item)

        self.btn_gmm.setEnabled(True)
        self.btn_gmm.setText('GMM run')
        
        # GMM 运行完成后，自动将 init ion freqs 重置为 {}
        self.linedit_ionfreqs.setText('{}')

    def result_clear(self):
        self.btn_clear.setEnabled(False)
        self.df_data['ion'] = ''
        self.df_data['harmonic'] = np.nan

        self.current_ion_brushes = None
        self.on_colorbar_levels_changed()

        if not hasattr(self, 'dynamic_labels'):
            self.dynamic_labels = []
        for old_text_item in self.dynamic_labels:
            self.p_main.removeItem(old_text_item)
        self.dynamic_labels.clear()

        self.btn_clear.setEnabled(True)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "select folder", './')
        if folder_path:
            self.linedit_csvFolder.setText(folder_path)

    def save_csv(self):
        self.btn_csvSave.setEnabled(False)
        self.btn_csvSave.setText('Saving .csv ...')
        try:
            # 匹配并增加 'm/q' 与 'gamma' 列
            if hasattr(self, 'df_ref') and self.df_ref is not None:
                if 'm/q' in self.df_ref.columns:
                    mq_map = dict(zip(self.df_ref['ion'], self.df_ref['m/q']))
                    self.df_data['m/q'] = self.df_data['ion'].map(mq_map)
                else:
                    self.df_data['m/q'] = np.nan

                if 'gamma' in self.df_ref.columns:
                    gamma_map = dict(zip(self.df_ref['ion'], self.df_ref['gamma']))
                    self.df_data['gamma'] = self.df_data['ion'].map(gamma_map)
                else:
                    self.df_data['gamma'] = np.nan
            else:
                self.df_data['m/q'] = np.nan
                self.df_data['gamma'] = np.nan

            new_order = [
                'peak_pos', 'err_pos', 'sigma', 'err_sigma', 'height_ratio', 
                'height_ion', 'exist_sate', 'exist_time', 'valid', 'pair_num', 
                'ion', 'harmonic', 'm/q', 'gamma', 'filename'
            ]
            
            cols_to_save = [c for c in new_order if c in self.df_data.columns]
            save_path = os.path.join(self.linedit_csvFolder.text().strip(), self.linedit_filename.text().strip())
            self.df_data[cols_to_save].to_csv(save_path, index=False)

            # 保存成功后弹出询问框
            reply = QMessageBox.question(
                self,
                'Plot Request',
                'Save successful! Do you want to plot the sigma f/f plot?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.show_sigma_plot()

        except Exception as e:
            QMessageBox.critical(
                self,
                "File saving error",
                f"Error: {e}\n\nPlease check and input file info again.",
                QMessageBox.Ok
            )
            self.btn_csvSave.setEnabled(True)
            self.btn_csvSave.setText('Save .csv')
            return

        self.btn_csvSave.setEnabled(True)
        self.btn_csvSave.setText('Save .csv')

    def show_sigma_plot(self):
        df_valid = self.df_data[self.df_data['ion'].notna() & (self.df_data['ion'] != '')].copy()
        if df_valid.empty:
            QMessageBox.information(self, "Notice", "No classified ion found in data!")
            return

        df_valid['f_i'] = df_valid['peak_pos'] / df_valid['harmonic']

        records = []
        for ion, group in df_valid.groupby('ion'):
            if len(group) == 0:
                continue
            
            f_mean = group['f_i'].mean()
            f_std = group['f_i'].std(ddof=1) if len(group) > 1 else 0.0
            rel_sigma = f_std / f_mean if f_mean != 0 else 0.0
            
            mq_val = group['m/q'].iloc[0] if 'm/q' in group.columns else np.nan
            gamma_val = group['gamma'].iloc[0] if 'gamma' in group.columns else np.nan

            records.append({
                'ion': ion,
                'm/q': mq_val,
                'gamma': gamma_val,
                'rel_sigma': rel_sigma
            })

        summary_df = pd.DataFrame(records).dropna(subset=['m/q', 'gamma'])

        if summary_df.empty:
            QMessageBox.warning(self, "Warning", "'m/q' or 'gamma' values could not be matched for any ion. Please check your ref file!")
            return

        self.sigma_win = SigmaPlotWindow(summary_df)
        self.sigma_win.show()

    def update_histograms(self):
        view_range = self.p_main.viewRange()
        x_range, y_range = view_range[0], view_range[1]
        
        y_label_pos = y_range[1] - (y_range[1] - y_range[0]) * 0.05

        for x_val, text_item in self.label_items:
            text_item.setPos(x_val, y_label_pos)
            if x_val > x_range[0] and x_val < x_range[1]:
                text_item.setVisible(True)
            else:
                text_item.setVisible(False)

        mask = (self.df_data['peak_pos'] >= x_range[0]) & (self.df_data['peak_pos'] <= x_range[1]) & \
               (self.df_data['height_ion'] >= y_range[0]) & (self.df_data['height_ion'] <= y_range[1])
        visible_data = self.df_data[mask]

        self.p_hist_x.clear()
        self.p_hist_y.clear()

        if len(visible_data) >= 2:
            xb = self.spin_x_bins.value()
            xc, xe = np.histogram(visible_data['peak_pos'], bins=xb if xb > 0 else 'auto', range=x_range)
            bg_x = pg.BarGraphItem(x=(xe[:-1]+xe[1:])/2, height=xc, width=np.diff(xe), 
                                   brush=(100, 100, 200, 150), pen=(255,255,255,50))
            self.p_hist_x.addItem(bg_x)

            yb = self.spin_y_bins.value()
            yc, ye = np.histogram(visible_data['height_ion'], bins=yb if yb > 0 else 'auto', range=y_range)
            bg_y = pg.BarGraphItem(x0=0, y=(ye[:-1]+ye[1:])/2, width=yc, height=np.diff(ye), 
                                   brush=(200, 100, 100, 150), pen=(255,255,255,50))
            self.p_hist_y.addItem(bg_y)
            self.p_hist_y.setXRange(0, max(yc) * 1.1 if len(yc)>0 else 10)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = FastLargeDataPlotter()
    demo.show()
    sys.exit(app.exec_())
