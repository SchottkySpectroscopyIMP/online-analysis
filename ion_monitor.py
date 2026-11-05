#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys, ast, os, platform, re
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QStatusBar, 
                            QPushButton, QMessageBox, QDoubleSpinBox, QLineEdit, QFileDialog,
                             QHBoxLayout, QSpinBox, QLabel, QGroupBox, QGridLayout, QCompleter, QRadioButton, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QStringListModel
from PyQt5.QtGui import QFont
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
import warnings

# 过滤 loky/joblib 关于物理核心数检测的 UserWarning 警告
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', message='.*Could not find the number of physical cores.*')

# 内置常用元素周期表 (Element -> Z)
ELEMENT_Z_TABLE = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

class Worker(QObject):
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

        # 1. 区域内没有数据点，直接返回原值
        if len(filtered_data) == 0:
            self.result_ready.emit(self.df['ion'].values, self.df['harmonic'].values)
            return

        # 2. 【核心优化】：若只传入了 1 种离子，直接将框选范围内的所有点强行赋值为此离子，不运行 GMM
        if len(x_init_list) == 1:
            self.df.loc[mask, 'ion'] = labels_name[0]
            self.df.loc[mask, 'harmonic'] = labels_harmonic[0]
            self.result_ready.emit(self.df['ion'].values, self.df['harmonic'].values)
            return

        # 3. 正常运行多离子 GMM 拟合
        y_init = np.mean(filtered_data[:,1])
        inital_means = np.array([[x, y_init] for x in x_init_list])

        gmm = GaussianMixture(n_components=len(x_init_list), means_init=inital_means, random_state=42)
        gmm.fit(filtered_data)

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


class SigmaPlotWindow(QMainWindow):
    def __init__(self, summary_df):
        super().__init__()
        self.setWindowTitle("σf/f vs m/q & γ")
        self.resize(1000, 600)
        self.summary_df = summary_df
        self.apply_global_font()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.coord_label = QLabel("m/q: 0.0000, γ: 0.0000, σf/f: 0.000000")
        self.status_bar.addPermanentWidget(self.coord_label)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.plot_widget = pg.PlotWidget()
        main_layout.addWidget(self.plot_widget, stretch=4)

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

        self.mq_min = self.summary_df['m/q'].min()
        self.mq_max = self.summary_df['m/q'].max()
        self.gamma_min = self.summary_df['gamma'].min()
        self.gamma_max = self.summary_df['gamma'].max()

        if self.mq_max == self.mq_min:
            self.mq_max += 1.0
        if self.gamma_max == self.gamma_min:
            self.gamma_max += 1.0

        self.vb_top = pg.ViewBox()
        self.p_main.scene().addItem(self.vb_top)
        self.axis_top = pg.AxisItem('top')
        self.axis_top.setLabel('γ')
        self.p_main.layout.addItem(self.axis_top, 1, 1)
        self.axis_top.linkToView(self.vb_top)
        self.vb_top.setYLink(self.p_main.vb)

        def update_views():
            self.vb_top.setGeometry(self.p_main.vb.sceneBoundingRect())

        def update_top_x_range():
            x0, x1 = self.p_main.vb.viewRange()[0]
            g0 = self.mq_to_gamma(x0)
            g1 = self.mq_to_gamma(x1)
            self.vb_top.setXRange(g0, g1, padding=0)

        self.p_main.vb.sigResized.connect(update_views)
        self.p_main.vb.sigXRangeChanged.connect(update_top_x_range)
        self.p_main.scene().sigMouseMoved.connect(self.on_mouse_moved)

        self.plot_data()

    def mq_to_gamma(self, mq):
        return self.gamma_min + (mq - self.mq_min) * (self.gamma_max - self.gamma_min) / (self.mq_max - self.mq_min)

    def gamma_to_mq(self, gamma):
        return self.mq_min + (gamma - self.gamma_min) * (self.mq_max - self.mq_min) / (self.gamma_max - self.gamma_min)

    def plot_data(self):
        if self.summary_df.empty:
            return

        mq_vals = self.summary_df['m/q'].values
        y_vals = self.summary_df['rel_sigma'].values
        ions = self.summary_df['ion'].values

        scatter_bottom = pg.ScatterPlotItem(
            x=mq_vals, y=y_vals, size=12, symbol='o', 
            brush=pg.mkBrush('cyan'), pen=pg.mkPen('w', width=1)
        )
        self.p_main.addItem(scatter_bottom)

        for x, y, ion in zip(mq_vals, y_vals, ions):
            text_item = pg.TextItem(text=str(ion), color='yellow', anchor=(0.5, 1.2))
            text_item.setPos(x, y)
            self.p_main.addItem(text_item)

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

            def fit_func(x, g_t, d_brp, const):
                return np.abs(1.0 / (x**2) - 1.0 / (g_t**2)) * d_brp * 0.01 + const

            popt, _ = curve_fit(fit_func, x_data, y_data, p0=[init_gammat, init_dbrp, 0.0], maxfev=5000)
            fit_g_t, fit_d_brp, fit_const = popt[0], popt[1], popt[2]

            self.spin_gammat.setValue(float(fit_g_t))
            self.spin_dbrp.setValue(float(fit_d_brp))

            if hasattr(self, 'fit_curve') and self.fit_curve in self.p_main.items:
                self.p_main.removeItem(self.fit_curve)

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
        self.label_items = []  
        self.ref_lines_and_texts = [] 
        self.files_loaded = False
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

        self.p_main = self.win.addPlot(row=0, col=0)
        self.p_main.showGrid(x=True, y=True, alpha=0.3)
        self.p_main.setLabel('left', 'Height Ion')
        
        self.p_hist_y = self.win.addPlot(row=0, col=1)
        self.p_hist_y.setFixedWidth(150)
        self.p_hist_y.setYLink(self.p_main)
        self.p_hist_y.getAxis('left').setStyle(showValues=False)

        self.p_hist_x = self.win.addPlot(row=1, col=0)
        self.p_hist_x.setFixedHeight(150)
        self.p_hist_x.setXLink(self.p_main)

        file_group = QGroupBox("Data file Control")
        file_layout = QGridLayout()
        file_layout.setVerticalSpacing(4)

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
        self.linedit_refIons.setPlaceholderText("{'195Os75(1)+': [308000756.01, 204, 324.786], ...}")
        file_layout.addWidget(self.linedit_refIons, 3, 1)

        self.btn_addRef = QPushButton('Add ref.')
        file_layout.addWidget(self.btn_addRef, 4, 1)

        file_layout.addWidget(QLabel('Ion:'), 5, 0)
        self.linedit_search_ref_ion = QLineEdit()
        self.linedit_search_ref_ion.setPlaceholderText("e.g. 206Tl79(0)+")
        self.ref_ion_completer = QCompleter(self)
        self.ref_ion_completer.setFilterMode(Qt.MatchContains)
        self.ref_ion_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.linedit_search_ref_ion.setCompleter(self.ref_ion_completer)
        file_layout.addWidget(self.linedit_search_ref_ion, 5, 1)

        self.btn_search_ref_ion = QPushButton('Search ion')
        file_layout.addWidget(self.btn_search_ref_ion, 6, 1)

        self.textedit_ref_info = QTextEdit()
        self.textedit_ref_info.setReadOnly(True)
        self.textedit_ref_info.setLineWrapMode(QTextEdit.NoWrap)
        self.textedit_ref_info.setFixedHeight(75)
        file_layout.addWidget(self.textedit_ref_info, 7, 0, 1, 2)

        self.p_hist_q = pg.PlotWidget()
        self.p_hist_q.setFixedHeight(105)
        self.p_hist_q.showGrid(x=True, y=True, alpha=0.3)
        self.p_hist_q.setLabel('bottom', 'Q')
        self.p_hist_q.setLabel('left', 'Count')
        file_layout.addWidget(self.p_hist_q, 8, 0, 1, 2)

        self.line_q_min = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('r', width=2))
        self.line_q_max = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('r', width=2))
        self.p_hist_q.addItem(self.line_q_min)
        self.p_hist_q.addItem(self.line_q_max)

        q_control_layout = QHBoxLayout()
        q_control_layout.setContentsMargins(0, 0, 0, 0)
        self.radio_q_filter = QRadioButton("Enable Q Filter")
        self.radio_q_filter.setChecked(True)

        self.spin_q_min = QSpinBox()
        self.spin_q_min.setRange(0, 200)
        self.spin_q_max = QSpinBox()
        self.spin_q_max.setRange(0, 200)

        q_control_layout.addWidget(self.radio_q_filter)
        q_control_layout.addWidget(QLabel("Q min:"))
        q_control_layout.addWidget(self.spin_q_min)
        q_control_layout.addWidget(QLabel("Q max:"))
        q_control_layout.addWidget(self.spin_q_max)

        file_layout.addLayout(q_control_layout, 9, 0, 1, 2)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group, 0, 1)

        self.btn_dataFile.clicked.connect(self.select_dataFile)
        self.btn_refFile.clicked.connect(self.select_refFlie)
        self.btn_fileLoad.clicked.connect(self.load_data)
        self.btn_addRef.clicked.connect(self.add_refIons)
        self.btn_search_ref_ion.clicked.connect(self.search_ref_ion_info)

        self.line_q_min.sigPositionChanged.connect(self.on_q_line_min_dragged)
        self.line_q_max.sigPositionChanged.connect(self.on_q_line_max_dragged)
        self.spin_q_min.valueChanged.connect(self.on_q_spin_min_changed)
        self.spin_q_max.valueChanged.connect(self.on_q_spin_max_changed)
        self.radio_q_filter.toggled.connect(self.on_q_filter_toggled)

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
        self.p_main.sigRangeChanged.connect(self.update_histograms)

        gmmcontrol_group = QGroupBox("GMM Control")
        gmmcontrol_layout = QGridLayout()

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

        layout.setColumnStretch(0, 4)
        layout.setColumnStretch(1, 1)

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
            
            for col in ['ion', 'harmonic', 'm/q', 'gamma']:
                if col not in self.df_data.columns:
                    if col == 'ion':
                        self.df_data['ion'] = ''
                    else:
                        self.df_data[col] = np.nan
            
            self.df_data['ion'] = self.df_data['ion'].fillna('').astype(str)
            
            if hasattr(self, 'df_ref') and self.df_ref is not None and not self.df_ref.empty:
                for ref_col in ['harmonic', 'm/q', 'gamma']:
                    if ref_col in self.df_ref.columns:
                        ref_map = dict(zip(self.df_ref['ion'], self.df_ref[ref_col]))
                        mapped_series = self.df_data['ion'].map(ref_map)
                        self.df_data[ref_col] = self.df_data[ref_col].combine_first(mapped_series)
            
            self.df_ref.columns = self.df_ref.columns.str.replace(r"\s*\(.*?\)", "", regex=True)
            self.current_ion_brushes = None

            self.refresh_completers()
            self.plot_reference_lines()
            self.plot_q_histogram()

            self.plot_data()
            self.update_histograms()

            if (self.df_data['ion'] != '').any():
                self.update_ion_visualization()

            self.linedit_filename.setText(self.data_name)
            self.files_loaded = True
        except Exception as e:
            QMessageBox.critical(
                self,
                "File loading error",
                f"Error: {e}\n\nPlease check and select files again.",
                QMessageBox.Ok
            )
            self.btn_fileLoad.setEnabled(True)
            self.btn_fileLoad.setText('Load files')
            self.files_loaded = False
            return
        self.btn_fileLoad.setEnabled(True)
        self.btn_fileLoad.setText('Load files')

    def refresh_completers(self):
        if not hasattr(self, 'df_ref') or self.df_ref is None or self.df_ref.empty:
            return

        if 'ion' in self.df_ref.columns and 'harmonic' in self.df_ref.columns:
            ref_items = []
            for _, row in self.df_ref.dropna(subset=['ion', 'harmonic']).iterrows():
                ref_items.append(f"{row['ion']}, {int(row['harmonic'])}")
            self.ion_completer.setModel(QStringListModel(ref_items, self))

        if 'ion' in self.df_ref.columns:
            unique_ions = sorted(list(self.df_ref['ion'].dropna().astype(str).unique()))
            self.ref_ion_completer.setModel(QStringListModel(unique_ions, self))

    def add_refIons(self):
        if not getattr(self, 'files_loaded', False) or self.df_ref is None or self.df_data is None:
            QMessageBox.warning(self, "Warning", "Please load files first before adding reference ions!")
            return

        self.btn_addRef.setEnabled(False)
        self.btn_addRef.setText('Adding ions ...')

        try:
            input_text = self.linedit_refIons.text().strip()
            if not input_text:
                QMessageBox.warning(self, "Warning", "Please input reference ions dict!")
                self.btn_addRef.setEnabled(True); self.btn_addRef.setText('Add ref.')
                return

            list_vals = ast.literal_eval(input_text)
            if not isinstance(list_vals, dict):
                raise ValueError("Input format must be a python dict.")

            data_f_min = self.df_data['peak_pos'].min()
            data_f_max = self.df_data['peak_pos'].max()

            q_col = next((c for c in ['Q', 'q'] if c in self.df_ref.columns), 'Q')

            new_rows = []
            replaced_ions = []
            added_ions = []

            for ion_str, ion_info in list_vals.items():
                if (self.df_ref['ion'].astype(str) == ion_str).any():
                    replaced_ions.append(ion_str)
                    self.df_ref = self.df_ref[self.df_ref['ion'].astype(str) != ion_str]
                else:
                    added_ions.append(ion_str)

                f_hz = float(ion_info[0])
                h_in = int(ion_info[1])
                sig_hz = float(ion_info[2])

                match = re.match(r"^(\d+)([A-Za-z]+)(\d+)", ion_str)
                if not match:
                    QMessageBox.critical(self, "Ion Format Error", f"Cannot parse ion pattern '{ion_str}'!\nExpected format like '195Os75(1)+'.")
                    self.btn_addRef.setEnabled(True); self.btn_addRef.setText('Add ref.')
                    return

                a_val = int(match.group(1))
                element_symbol = match.group(2)
                q_val = int(match.group(3))

                if element_symbol not in ELEMENT_Z_TABLE:
                    QMessageBox.critical(self, "Element Error", f"Unknown element symbol '{element_symbol}' in ion '{ion_str}'!\nOperation cancelled.")
                    self.btn_addRef.setEnabled(True); self.btn_addRef.setText('Add ref.')
                    return

                z_val = ELEMENT_Z_TABLE[element_symbol]
                n_val = a_val - z_val

                diff = z_val - q_val
                type_map = {0: 'bare', 1: 'H-like', 2: 'He-like', 3: 'Li-like', 4: 'Be-like', 5: 'B-like'}
                type_str = type_map.get(diff, '')

                rev_freq_mhz = f_hz / (h_in * 1e6)

                min_h = int(np.ceil(data_f_min / (rev_freq_mhz * 1e6)))
                max_h = int(np.floor(data_f_max / (rev_freq_mhz * 1e6)))
                valid_harmonics = range(min_h, max_h + 1) if min_h <= max_h else [h_in]

                for h in valid_harmonics:
                    p_loc_khz = (h * rev_freq_mhz * 1e6) / 1000.0
                    p_sig_khz = (sig_hz * (h / h_in)) / 1000.0

                    row_dict = {
                        'ion': ion_str,
                        'element': element_symbol,
                        q_col: q_val,
                        'N': n_val,
                        'Z': z_val,
                        'peak_loc': p_loc_khz,
                        'peak_sig': p_sig_khz,
                        'harmonic': h,
                        'rev_freq': rev_freq_mhz,
                        'half_life': np.nan,
                        'yield': np.nan,
                        'type': type_str,
                        'weight': np.nan,
                        'peak_max': np.nan,
                        'source': 'new',
                        'm/q': a_val / q_val if q_val != 0 else np.nan
                    }
                    new_rows.append(row_dict)

            df_new = pd.DataFrame(new_rows)
            self.df_ref = pd.concat([self.df_ref, df_new], ignore_index=True)

            self.refresh_completers()
            self.plot_reference_lines()
            self.plot_q_histogram()

            msg = "Successfully processed reference ions!\n"
            if replaced_ions:
                msg += f"\n• Replaced existing ion(s):\n  {', '.join(replaced_ions)}"
            if added_ions:
                msg += f"\n• Added new ion(s):\n  {', '.join(added_ions)}"

            QMessageBox.information(self, "Success", msg)
            self.linedit_refIons.clear()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Input format error",
                f"Failed to add ref ions!\n\nError: {e}\n\nPlease check format example:\n{{'195Os75(1)+':[308000756.01, 204, 324.786]}}",
                QMessageBox.Ok
            )
        finally:
            self.btn_addRef.setEnabled(True)
            self.btn_addRef.setText('Add ref.')

    def plot_reference_lines(self):
        if hasattr(self, 'ref_lines_and_texts'):
            for line, text, _, _ in self.ref_lines_and_texts:
                if line in self.p_main.items:
                    self.p_main.removeItem(line)
                if text in self.p_main.items:
                    self.p_main.removeItem(text)

        self.ref_lines_and_texts = []
        self.label_items = []

        if not hasattr(self, 'df_ref') or self.df_ref is None or self.df_ref.empty:
            return

        q_col = next((c for c in ['Q', 'q'] if c in self.df_ref.columns), None)

        y_range = self.p_main.viewRange()[1]
        y_label_pos = y_range[1] - (y_range[1] - y_range[0]) * 0.05

        for _, row in self.df_ref.iterrows():
            x_val = row['peak_loc'] * 1e3
            q_val = int(row[q_col]) if q_col and pd.notna(row[q_col]) else 0
            
            p_sig = row['peak_sig'] * 1e3 if pd.notna(row.get('peak_sig')) else 0.0
            ion_label = f"{str(row['ion'])}\nsigma: {p_sig:.3f} Hz\nh = {int(row['harmonic']):d}"
            
            line = pg.InfiniteLine(pos=x_val, angle=90, pen=pg.mkPen('lightgray', width=1.5, style=Qt.DashLine))
            self.p_main.addItem(line)

            text = pg.TextItem(text=ion_label, color='lightgray', anchor=(0, 0))
            text.setPos(x_val, y_label_pos)
            self.p_main.addItem(text)
            
            self.label_items.append((x_val, text))
            self.ref_lines_and_texts.append((line, text, q_val, x_val))

        self.dynamic_labels = []

    def plot_q_histogram(self):
        self.p_hist_q.clear()
        self.p_hist_q.addItem(self.line_q_min)
        self.p_hist_q.addItem(self.line_q_max)

        q_col = next((c for c in ['Q', 'q'] if c in self.df_ref.columns), None)
        if q_col is None:
            return

        df_unique_ions = self.df_ref.drop_duplicates(subset=['ion'])
        q_series = df_unique_ions[q_col].dropna().astype(int)

        if q_series.empty:
            return

        q_min, q_max = q_series.min(), q_series.max()

        q_counts = q_series.value_counts().sort_index()
        x_centers = np.arange(q_min, q_max + 1)
        y_counts = [q_counts.get(q, 0) for q in x_centers]

        bg_q = pg.BarGraphItem(x=x_centers, height=y_counts, width=0.8, brush=(120, 180, 250, 180), pen='w')
        self.p_hist_q.addItem(bg_q)
        self.p_hist_q.setXRange(q_min - 1, q_max + 1)

        self.spin_q_min.blockSignals(True)
        self.spin_q_max.blockSignals(True)
        self.spin_q_min.setRange(q_min, q_max)
        self.spin_q_max.setRange(q_min, q_max)
        self.spin_q_min.setValue(q_min)
        self.spin_q_max.setValue(q_max)
        self.spin_q_min.blockSignals(False)
        self.spin_q_max.blockSignals(False)

        self.line_q_min.setValue(q_min)
        self.line_q_max.setValue(q_max)

        self.radio_q_filter.setChecked(True)
        self.on_q_filter_toggled(True)

    def on_q_line_min_dragged(self):
        val = int(round(self.line_q_min.value()))
        self.spin_q_min.blockSignals(True)
        self.spin_q_min.setValue(val)
        self.spin_q_min.blockSignals(False)

    def on_q_line_max_dragged(self):
        val = int(round(self.line_q_max.value()))
        self.spin_q_max.blockSignals(True)
        self.spin_q_max.setValue(val)
        self.spin_q_max.blockSignals(False)

    def on_q_spin_min_changed(self, val):
        self.line_q_min.setValue(val)

    def on_q_spin_max_changed(self, val):
        self.line_q_max.setValue(val)

    def on_q_filter_toggled(self, checked):
        self.spin_q_min.setEnabled(not checked)
        self.spin_q_max.setEnabled(not checked)
        self.line_q_min.setMovable(not checked)
        self.line_q_max.setMovable(not checked)

        self.update_main_plot_ref_lines()

    def update_main_plot_ref_lines(self):
        if not hasattr(self, 'ref_lines_and_texts'):
            return

        is_filtered = self.radio_q_filter.isChecked()
        q_start = self.spin_q_min.value()
        q_end = self.spin_q_max.value()
        x_range = self.p_main.viewRange()[0]

        for line, text, q_val, x_val in self.ref_lines_and_texts:
            in_x_view = (x_range[0] <= x_val <= x_range[1])

            if is_filtered:
                if (q_start <= q_val <= q_end) and in_x_view:
                    line.setVisible(True)
                    text.setVisible(True)
                else:
                    line.setVisible(False)
                    text.setVisible(False)
            else:
                line.setVisible(in_x_view)
                text.setVisible(in_x_view)

    def search_ref_ion_info(self):
        ion_name = self.linedit_search_ref_ion.text().strip()
        if not ion_name:
            return

        if not hasattr(self, 'df_ref') or self.df_ref is None or self.df_ref.empty:
            QMessageBox.warning(self, "Warning", "Please load reference file first!")
            return

        matched = self.df_ref[self.df_ref['ion'].astype(str) == ion_name]
        if matched.empty:
            QMessageBox.warning(self, "Warning", f"Ion '{ion_name}' not found in reference file!")
            return

        row_0 = matched.iloc[0]
        
        def get_val(row, possible_keys, default="N/A"):
            for k in possible_keys:
                if k in row.index and pd.notna(row[k]):
                    return row[k]
            return default

        weight_val = get_val(row_0, ['weight', 'Weight'])
        yield_val = get_val(row_0, ['yield', 'Yield'])
        rev_freq_val = get_val(row_0, ['rev_freq', 'rev freq', 'RevFreq', 'rev_frequency'])
        half_life_val = get_val(row_0, ['half_life', 'half life', 'HalfLife', 'halflife'])

        try:
            w_str = f"{float(weight_val):.3e}" if weight_val != "N/A" else "N/A"
        except (ValueError, TypeError):
            w_str = str(weight_val)

        try:
            y_str = f"{float(yield_val):.3e}" if yield_val != "N/A" else "N/A"
        except (ValueError, TypeError):
            y_str = str(yield_val)

        try:
            rf_str = f"{float(rev_freq_val):.6f}" if rev_freq_val != "N/A" else "N/A"
        except (ValueError, TypeError):
            rf_str = str(rev_freq_val)

        hl_str = str(half_life_val)

        res_text = f"{ion_name} in ref. file:\n"
        res_text += f"weight: {w_str}, yield: {y_str}, rev freq: {rf_str} MHz, half life: {hl_str}\n"

        for _, row in matched.iterrows():
            h_val = get_val(row, ['harmonic', 'Harmonic'], 0)
            p_loc_val = get_val(row, ['peak_loc', 'peak_pos', 'freq'], 0.0)
            p_max_val = get_val(row, ['peak_max', 'height_ion', 'height', 'max_peak'], 0.0)

            try:
                h_int = int(h_val)
            except (ValueError, TypeError):
                h_int = 0

            try:
                p_loc_flt = float(p_loc_val)
            except (ValueError, TypeError):
                p_loc_flt = 0.0

            try:
                p_max_flt = float(p_max_val)
            except (ValueError, TypeError):
                p_max_flt = 0.0

            res_text += f"harmonic: {h_int:d}, peak_loc: {p_loc_flt:.3f} kHz, peak_max: {p_max_flt:.3e}\n"

        self.textedit_ref_info.setPlainText(res_text.strip())

    def update_ion_visualization(self):
        # 1. 彻底清理主图上所有旧的动态离子标签 (TextItem)
        if hasattr(self, 'dynamic_labels') and self.dynamic_labels:
            for old_text_item in self.dynamic_labels:
                if old_text_item in self.p_main.items:
                    self.p_main.removeItem(old_text_item)
            self.dynamic_labels.clear()
        else:
            self.dynamic_labels = []

        if not hasattr(self, 'df_data') or self.df_data is None or self.df_data.empty:
            return

        # 2. 仅提取当前 df_data 中【实际已经被标记】的有效离子列表
        real_unique_ions = sorted([str(ion).strip() for ion in self.df_data['ion'].unique() if str(ion).strip() != ''])

        # 3. 构建高对比度离散颜色映射表（黄金分割步长，保证相邻索引颜色极大区分）
        ion_color_map = {}
        GOLDEN_RATIO_CONJUGATE = 0.618033988749895

        for i, ion_str in enumerate(real_unique_ions):
            hue = int(((i * GOLDEN_RATIO_CONJUGATE) % 1.0) * 360.0)
            color = pg.mkColor('w')
            color.setHsv(hue, 255, 255)
            ion_color_map[ion_str] = color

        # 4. 刷新数据点的填充颜色刷 (Brush)
        self.current_ion_brushes = []
        for ion in self.df_data['ion']:
            ion_str = str(ion).strip()
            if ion_str != '' and ion_str in ion_color_map:
                self.current_ion_brushes.append(pg.mkBrush(ion_color_map[ion_str]))
            else:
                self.current_ion_brushes.append(None)

        # 5. 刷新散点图表颜色的呈现
        self.on_colorbar_levels_changed()

        # 6. 重新绘制离子标签（核心改进：谐波标准化清洗与重叠中心合并）
        for ion_name in real_unique_ions:
            df_sub = self.df_data[self.df_data['ion'] == ion_name].copy()
            if df_sub.empty:
                continue

            ion_color = ion_color_map.get(ion_name, pg.mkColor('w'))

            # 将 harmonic 统一安全转换为纯整数 int
            harmonics_clean = []
            for h in df_sub['harmonic'].dropna().values:
                try:
                    harmonics_clean.append(int(round(float(h))))
                except (ValueError, TypeError):
                    pass
            unique_h_list = sorted(list(set(harmonics_clean)))

            # 如果解析出了有效的不同谐波分组
            if len(unique_h_list) > 0:
                for h in unique_h_list:
                    # 使用数值容差匹配，解决 float vs int 的不匹配问题
                    _df_sub = df_sub[df_sub['harmonic'].apply(
                        lambda x: pd.notna(x) and int(round(float(x))) == h if str(x).replace('.','',1).isdigit() else False
                    )]
                    if _df_sub.empty:
                        continue
                    
                    x_center = _df_sub['peak_pos'].mean()
                    y_position = _df_sub['height_ion'].min()

                    text_item = pg.TextItem(text=str(ion_name), color=ion_color, anchor=(0.5, 0))
                    text_item.setPos(x_center, y_position)
                    self.p_main.addItem(text_item)
                    self.dynamic_labels.append(text_item)
            else:
                # 若无任何谐波标定，整个离子团只绘制一个整体中心标签
                x_center = df_sub['peak_pos'].mean()
                y_position = df_sub['height_ion'].min()

                text_item = pg.TextItem(text=str(ion_name), color=ion_color, anchor=(0.5, 0))
                text_item.setPos(x_center, y_position)
                self.p_main.addItem(text_item)
                self.dynamic_labels.append(text_item) 

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
        
        # 保证 harmonic 为干净的 int/NaN 混合类型
        cleaned_h = []
        for val in possible_harmonic_range:
            try:
                cleaned_h.append(int(round(float(val))))
            except (ValueError, TypeError):
                cleaned_h.append(np.nan)
                
        self.df_data['harmonic'] = cleaned_h
        
        self.update_ion_visualization()
        self.btn_gmm.setEnabled(True)
        self.btn_gmm.setText('GMM run')
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
            if hasattr(self, 'df_ref') and self.df_ref is not None:
                if 'm/q' in self.df_ref.columns:
                    mq_map = dict(zip(self.df_ref['ion'], self.df_ref['m/q']))
                    matched_mq = self.df_data['ion'].map(mq_map)
                    self.df_data['m/q'] = matched_mq.combine_first(self.df_data.get('m/q', pd.Series(dtype=float)))
                elif 'm/q' not in self.df_data.columns:
                    self.df_data['m/q'] = np.nan

                if 'gamma' in self.df_ref.columns:
                    gamma_map = dict(zip(self.df_ref['ion'], self.df_ref['gamma']))
                    matched_gamma = self.df_data['ion'].map(gamma_map)
                    self.df_data['gamma'] = matched_gamma.combine_first(self.df_data.get('gamma', pd.Series(dtype=float)))
                elif 'gamma' not in self.df_data.columns:
                    self.df_data['gamma'] = np.nan
            else:
                for col in ['m/q', 'gamma']:
                    if col not in self.df_data.columns:
                        self.df_data[col] = np.nan

            save_path = os.path.join(self.linedit_csvFolder.text().strip(), self.linedit_filename.text().strip())
            self.df_data.to_csv(save_path, index=False)

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

        self.update_main_plot_ref_lines()

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
