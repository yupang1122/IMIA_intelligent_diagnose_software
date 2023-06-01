# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
# from sklearn import feature_selection
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import Normalizer
# from sklearn.preprocessing import normalize
import sys, random

from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QWidget, QFileDialog

from PyQt5.QtCore import Qt, pyqtSlot

from PyQt5.QtGui import QStandardItemModel, QStandardItem, QPainter, QPen, QPixmap

from PyQt5.QtChart import *

from chart_gui.dialogwindow import Ui_Dialog


class QmyMainWindow(QDialog):
    COL_NAME = 0  # 姓名的列编号
    COL_MATH = 1  # 数学的列编号
    COL_CHINESE = 2  # 语文的列编号
    COL_ENGLISH = 3  # 英语的列编号
    COL_AVERAGE = 4  # 平均分的列编号

    def __init__(self, rect_cell_result, svs_name,parent=None, ):
        super(QmyMainWindow, self).__init__(parent)  # 调用父类构造函数，创建窗体
        self.ui = Ui_Dialog()  # 创建UI对象
        self.ui.setupUi(self)  # 构造UI界面
        self.rect_cell_result = rect_cell_result
        self.survive_path = '../dataprocess/survive_data/'
        print(self.rect_cell_result)
        self.setWindowTitle("图像处理结果")
        index_path = os.path.join('../dataprocess/data-index/', svs_name, str(224))  ##背景筛选信息位置 创建这个文件夹
        background_index = index_path + '/' + 'svs_index.txt'
        print(background_index)
        self.background_index = background_index
        self.ui.tableView.setAlternatingRowColors(True)
        self.ui.treeWidget.setAlternatingRowColors(True)
        self.setStyleSheet("QTreeWidget, QTableView{"
                           "alternate-background-color:rgb(170, 241, 190)}")

        self.__studCount = 9  # 学生人数
        self.ui.spinCount.setValue(self.__studCount)  ##人数标框

        self.dataModel = QStandardItemModel(self)  # 数据模型
        self.ui.tableView.setModel(self.dataModel)  # 设置数据模型
        self.dataModel.itemChanged.connect(self.do_calcuAverage)  # 自动计算平均分

        self.__generateData()  # 初始化数据
        self.__surveyData()  # 数据统计

        self.__iniBarChart()  # 柱状图初始化+
        # self.__iniStackedBar()  # 堆积图初始化
        self.iniCoxRes()
        # self.__iniPercentBar()     #百分比图初始化
        # self.__iniPieChart()       #饼图初始化

    ##  ==============自定义功能函数========================
    def __generateData(self):  ##随机生成分数数据
        self.dataModel.clear()
        headerList = ["类型", "有效切片数"]
        self.dataModel.setHorizontalHeaderLabels(headerList)  # 设置表头文字
        stuname = ["脂肪","背景","碎片","淋巴","肌肉","粘液","正常","基质","肿瘤"]
        scorenum = [0,0,0,0,0,0,0,0,0]
        with open(self.background_index, 'r') as f:
            annotations = f.readlines()
        print("ok")
        rate1 = []
        rate2 = []
        rate3 = []
        rate4 = []
        rate5 = []
        rate6 = []
        rate7 = []
        rate8 = []
        rate9 = []
        maxrate = [rate1,rate2,rate3,rate4,rate5,rate6,rate7,rate8,rate9]
        for annotation in annotations:
            annotation = annotation.strip().split(' ')

            max_rate = float(int(annotation[1]) / 255)
            # max_rate = int(annotation[1])

            max_type = int(annotation[4])

            if max_type != 0:
                maxrate[max_type-1].append(max_rate)

                scorenum[max_type-1] += 1


        print("ooook")
        print(scorenum)

        finalrate = []
        avgrate = []
        for k in range(9):
            if scorenum[k] != 0:
            # # print(max(enumerate(maxrate[k])))
            # print(maxrate[k])
                finalrate.append(max(maxrate[k]))
                avgrate.append(float(sum(maxrate[k]) / len(maxrate[k])))
            else:
                finalrate.append(0)
                avgrate.append(0)
        # finalnum = [len(rate1),len(rate2),len(rate3),len(rate4),len(rate5),len(rate6),len(rate7),len(rate8),len(rate9)]
        # finalrate = [max(rate1),max(rate2),max(rate3),max(rate4),max(rate5),max(rate6),max(rate7),max(rate8),max(rate9)]
        #
        # avgrate = [sum(rate1)/len(rate1),sum(rate2)/len(rate2),sum(rate3)/len(rate3),sum(rate4)/len(rate4),sum(rate5)/len(rate5),
        #            sum(rate6)/len(rate6),sum(rate7)/len(rate7),sum(rate8)/len(rate8),sum(rate9)/len(rate9)]
        print("ook")
        for i in range(self.__studCount):
            itemList = []
            # studName = "类型%2d" % (i + 1)
            studName = stuname[i]
            item = QStandardItem(studName)  # 创建item
            item.setTextAlignment(Qt.AlignHCenter)
            itemList.append(item)  # 添加到列表

            avgScore = 0
            # for j in range(self.COL_MATH, 1 + self.COL_ENGLISH):  # 类型
            pointt = [scorenum[i],finalrate[i],avgrate[i]]
            for j in range(1,2):  # 类型
                score = pointt[j-1]
                item = QStandardItem("%.0f" % score)  # 创建 item
                item.setTextAlignment(Qt.AlignHCenter)
                itemList.append(item)  # 添加到列表
                avgScore = avgScore + score

            # rate = 250.0 + random.randint(-5, 5)
            # item = QStandardItem("%.0f" % rate)  # 创建平均分item
            # item.setTextAlignment(Qt.AlignHCenter)
            # item.setFlags(item.flags() & (not Qt.ItemIsEditable))  # 平均分不允许编辑
            # itemList.append(item)  # 添加到列表
            self.dataModel.appendRow(itemList)  # 添加到数据模型

    def __surveyData(self):  ##统计各分数段人数
        if self.rect_cell_result == None:
            return
        for i in range(len(self.rect_cell_result)):
            # for j in range(len(self.rect_cell_result[i])):
            rect = self.rect_cell_result[i][0]
            cell_number = self.rect_cell_result[i][1]
            # mean = self.rect_cell_result[i][2][0]
            # max_min = [self.rect_cell_result[i][2][1], self.rect_cell_result[i][2][2]]
            item = self.ui.treeWidget.topLevelItem(i)
            item.setText(1, str(rect))
            item.setTextAlignment(1, Qt.AlignHCenter)

            item = self.ui.treeWidget.topLevelItem(i)
            item.setText(2, str(cell_number))
            item.setTextAlignment(2, Qt.AlignHCenter)

            # item = self.ui.treeWidget.topLevelItem(i)
            # item.setText(3, str(mean))
            # item.setTextAlignment(3, Qt.AlignHCenter)
            #
            # item = self.ui.treeWidget.topLevelItem(i)
            # item.setText(4, str(max_min))
            # item.setTextAlignment(4, Qt.AlignHCenter)

        #
        # for i in range(self.COL_MATH, 1+self.COL_ENGLISH): #统计 三列  先列后行的逻辑
        #    cnt50,cnt60,cnt70,cnt80,cnt90=0,0,0,0,0
        #    for j in range(self.dataModel.rowCount()):      #行数等于学生人数
        #       val=float(self.dataModel.item(j,i).text())   #分数
        #       if val<60:
        #          cnt50 =cnt50+1
        #       elif (val>=60 and val<70):
        #          cnt60 = cnt60+1
        #       elif (val>=70 and val<80):
        #          cnt70 =cnt70+1
        #       elif (val>=80 and val<90):
        #          cnt80 =cnt80+1
        #       else:
        #          cnt90 =cnt90+1
        #    print('ok')
        #    item=self.ui.treeWidget.topLevelItem(0)   #第1行,<60
        #    item.setText(i,str(cnt50))                #第i列
        #    item.setTextAlignment(i,Qt.AlignHCenter)
        #
        #    item=self.ui.treeWidget.topLevelItem(1)   #第2行,[60,70)
        #    item.setText(i,str(cnt60))  # 第i列
        #    item.setTextAlignment(i,Qt.AlignHCenter)
        #
        #    item=self.ui.treeWidget.topLevelItem(2)   #第3行,[70,80)
        #    item.setText(i,str(cnt70))  # 第i列
        #    item.setTextAlignment(i,Qt.AlignHCenter)
        #
        #    item=self.ui.treeWidget.topLevelItem(3)   #第4行,[80,90)
        #    item.setText(i,str(cnt80))  # 第i列
        #    item.setTextAlignment(i,Qt.AlignHCenter)
        #
        #    item=self.ui.treeWidget.topLevelItem(4)   #第5行,[90,100]
        #    item.setText(i,str(cnt90))  # 第i列
        #    item.setTextAlignment(i,Qt.AlignHCenter)

    def __iniBarChart(self):  ##初始化柱状图
        chart = QChart()

        ##      chart.setAnimationOptions(QChart.SeriesAnimations)
        self.ui.chartViewBar.setChart(chart)  # 为ChartView设置chart
        self.ui.chartViewBar.setRenderHint(QPainter.Antialiasing)
        self.ui.chartViewBar.setCursor(Qt.CrossCursor)  # 设置鼠标指针为十字星

    # def __iniStackedBar(self):  ##初始化堆叠柱状图
    #     chart = QChart()
    #
    #     ##      chart.setAnimationOptions(QChart.SeriesAnimations)
    #     self.ui.chartViewStackedBar.setChart(chart)  # 为ChartView设置chart
    #     self.ui.chartViewStackedBar.setRenderHint(QPainter.Antialiasing)
    #     self.ui.chartViewStackedBar.setCursor(Qt.CrossCursor)  # 设置鼠标指针为十字星
    #
    # def __iniPercentBar(self):  ##百分比柱状图初始化
    #     chart = QChart()
    #
    #     ##      chart.setAnimationOptions(QChart.SeriesAnimations)
    #     self.ui.chartViewPercentBar.setChart(chart)  # 为ChartView设置chart
    #     self.ui.chartViewPercentBar.setRenderHint(QPainter.Antialiasing)
    #     self.ui.chartViewPercentBar.setCursor(Qt.CrossCursor)  # 设置鼠标指针为十字星
    #
    # def __iniPieChart(self):  ##饼图初始化
    #     chart = QChart()
    #
    #     chart.setAnimationOptions(QChart.SeriesAnimations)
    #     ##      chart.setAcceptHoverEvents(True) # 接受Hover事件
    #     self.ui.chartViewPie.setChart(chart)  # 为ChartView设置chart
    #     self.ui.chartViewPie.setRenderHint(QPainter.Antialiasing)
    #     self.ui.chartViewPie.setCursor(Qt.CrossCursor)  # 设置鼠标指针为十字星

    def __getCurrentChart(self):  ##获取当前QChart对象
        page = self.ui.tabWidget.currentIndex()
        if page == 0:
            chart = self.ui.chartViewBar.chart()
        elif page == 1:
            chart = self.ui.chartViewStackedBar.chart()
        elif page == 2:
            chart = self.ui.chartViewPercentBar.chart()
        else:
            chart = self.ui.chartViewPie.chart()
        return chart

    ##  ==============event事件处理函数==========================

    ##  ==========由connectSlotsByName()自动连接的槽函数============
    ## 工具栏按钮的功能
    # @pyqtSlot()   ##重新生成数据
    # def on_toolBtn_GenData_clicked(self):
    #    self.__studCount=self.ui.spinCount.value()  #学生人数
    #    self.__generateData()
    #    self.__surveyData()
    #
    # @pyqtSlot()   ##重新统计
    # def on_toolBtn_Counting_clicked(self):
    #    self.__surveyData()

    @pyqtSlot(int)  ##设置图表主题
    def on_comboTheme_currentIndexChanged(self, index):
        chart = self.__getCurrentChart()
        chart.setTheme(QChart.ChartTheme(index))

    @pyqtSlot(int)  ##图表动画
    def on_comboAnimation_currentIndexChanged(self, index):
        chart = self.__getCurrentChart()
        chart.setAnimationOptions(QChart.AnimationOption(index))

    ## ======page 1,  柱状图===================
    @pyqtSlot()  ##绘制柱状图
    def on_btnBuildBarChart_clicked(self):
        self.draw_barChart()

    @pyqtSlot()  ##绘制水平柱状图
    def on_btnBuildBarChartH_clicked(self):
        self.draw_barChart(False)

    def draw_barChart(self, isVertical=True):  ##绘制柱状图，或水平柱状图
        chart = self.ui.chartViewBar.chart()
        chart.removeAllSeries()  # 删除所有序列
        chart.removeAxis(chart.axisX())  # 删除坐标轴
        chart.removeAxis(chart.axisY())  # 删除坐标轴
        if isVertical:
            chart.setTitle("Barchart 演示")
            chart.legend().setAlignment(Qt.AlignBottom)
        else:
            chart.setTitle("Horizontal Barchart 演示")
            chart.legend().setAlignment(Qt.AlignRight)

        setMath = QBarSet("切片数")  # QBarSet
        setChinese = QBarSet("核数")
        setEnglish = QBarSet("核面积")

        seriesLine = QLineSeries()  # QLineSeries序列用于显示平均分
        seriesLine.setName("最高概率")
        pen = QPen(Qt.red)
        pen.setWidth(2)
        seriesLine.setPen(pen)

        boxWhiskSeries = QBoxPlotSeries()
        boxWhiskSeries.setName('面积均值')  # 从文件中读取数据  

        seriesLine.setPointLabelsVisible(True)  # 数据点标签可见
        if isVertical:
            seriesLine.setPointLabelsFormat("@yPoint")  # 显示y数值标签
        else:
            seriesLine.setPointLabelsFormat("@xPoint")  # 显示x数值标签

        font = seriesLine.pointLabelsFont()
        font.setPointSize(10)
        font.setBold(True)
        seriesLine.setPointLabelsFont(font)

        stud_Count = self.dataModel.rowCount()
        nameList = []
        for i in range(stud_Count):  # 从数据模型获取数据
            item = self.dataModel.item(i, self.COL_NAME)
            nameList.append(item.text())  # 姓名,用作坐标轴标签

            item = self.dataModel.item(i, self.COL_MATH)
            setMath.append(float(item.text()))  # 数学

            item = self.dataModel.item(i, self.COL_CHINESE)
            setChinese.append(float(item.text()))  # 语文

            item = self.dataModel.item(i, self.COL_ENGLISH)
            setEnglish.append(float(item.text()))  # 英语

            item = self.dataModel.item(i, self.COL_AVERAGE)
            if isVertical:
                seriesLine.append(i, float(item.text()))  # 平均分,用于柱状图
            else:
                seriesLine.append(float(item.text()), i)  # 平均分，用于水平柱状图

            box = QBoxSet()
            box.setValue(QBoxSet.LowerExtreme, 200)  # 下边沿        
            box.setValue(QBoxSet.UpperExtreme, 300)  # 上边沿         
            box.setValue(QBoxSet.Median, 266)  #  中位数         
            box.setValue(QBoxSet.LowerQuartile, 222)  # 下四分位数        
            box.setValue(QBoxSet.UpperQuartile, 270)  # 上四分位数
            boxWhiskSeries.append(box)

        # 创建一个序列 QBarSeries, 并添加三个数据集
        if isVertical:
            seriesBar = QBarSeries()  # 柱状图
        else:
            seriesBar = QHorizontalBarSeries()  # 水平柱状图

        seriesBar.append(setMath)  # 添加数据集
        seriesBar.append(setChinese)
        seriesBar.append(setEnglish)
        seriesBar.setLabelsVisible(True)  # 数据点标签可见
        seriesBar.setLabelsFormat("@value")  # 显示数值标签
        seriesBar.setLabelsPosition(QAbstractBarSeries.LabelsCenter)  # 数据标签显示位置
        seriesBar.hovered.connect(self.do_barSeries_Hovered)  # hovered信号
        seriesBar.clicked.connect(self.do_barSeries_Clicked)  # clicked信号

        chart.addSeries(seriesBar)  # 添加柱状图序列
        chart.addSeries(seriesLine)  # 添加折线图序列
        chart.addSeries(boxWhiskSeries)

        ##学生姓名坐标轴
        axisStud = QBarCategoryAxis()
        axisStud.append(nameList)  # 添加横坐标文字列表
        axisStud.setRange(nameList[0], nameList[stud_Count - 1])  # 这只坐标轴范围

        # 数值型坐标轴
        axisValue = QValueAxis()
        axisValue.setRange(0, 350)
        axisValue.setTitleText("癌类分析结果")
        axisValue.setTickCount(6)
        axisValue.applyNiceNumbers()
        #    axisValue.setLabelFormat("%.0f") #标签格式
        #    axisY.setGridLineVisible(false)
        #    axisY.setMinorTickCount(4)
        if isVertical:
            chart.setAxisX(axisStud, seriesBar)  # seriesBar
            chart.setAxisY(axisValue, seriesBar)
            chart.setAxisX(axisStud, seriesLine)  # seriesLine
            chart.setAxisY(axisValue, seriesLine)
        else:
            chart.setAxisX(axisValue, seriesBar)  # seriesBar
            chart.setAxisY(axisStud, seriesBar)
            chart.setAxisY(axisStud, seriesLine)  # seriesLine
            chart.setAxisX(axisValue, seriesLine)

        for marker in chart.legend().markers():  # QLegendMarker类型列表
            marker.clicked.connect(self.do_LegendMarkerClicked)

    ##=========page 2. StackedBar=========
    # @pyqtSlot()   ## 绘制StackedBar
    # def on_btnBuildStackedBar_clicked(self):
    #    self.draw_stackedBar()
    #
    # @pyqtSlot()   ## 绘制水平StackedBar
    # def on_btnBuildStackedBarH_clicked(self):
    #    self.draw_stackedBar(False)

    #    def draw_stackedBar(self,isVertical=True):   #堆叠柱状图
    #       chart =self.ui.chartViewStackedBar.chart()
    #       chart.removeAllSeries()       #删除所有序列
    #       chart.removeAxis(chart.axisX())     #删除坐标轴
    #       chart.removeAxis(chart.axisY())
    #       if isVertical:    #堆叠柱状图
    #          chart.setTitle("StackedBar 演示")
    #          chart.legend().setAlignment(Qt.AlignBottom)
    #       else:             #水平堆叠柱状图
    #          chart.setTitle("Horizontal StackedBar 演示")
    #          chart.legend().setAlignment(Qt.AlignRight)
    #
    #       ##创建三门课程的数据集
    #       setMath  =  QBarSet("数学")
    #       setChinese= QBarSet("语文")
    #       setEnglish= QBarSet("英语")
    #
    #       stud_Count=self.dataModel.rowCount()
    #       nameList=[]             #学生姓名列表
    #       for i in range(stud_Count):
    #          item=self.dataModel.item(i,self.COL_NAME)    #姓名
    #          nameList.append(item.text())
    #
    #          item=self.dataModel.item(i,self.COL_MATH)    #数学
    #          setMath.append(float(item.text()))
    #
    #          item=self.dataModel.item(i,self.COL_CHINESE)  #语文
    #          setChinese.append(float(item.text()))
    #
    #          item=self.dataModel.item(i,self.COL_ENGLISH)  #英语
    #          setEnglish.append(float(item.text()))
    #
    #       ##创建序列
    #       if isVertical:
    #          seriesBar = QStackedBarSeries()
    #       else:
    #          seriesBar = QHorizontalStackedBarSeries()
    #
    #       seriesBar.append(setMath)
    #       seriesBar.append(setChinese)
    #       seriesBar.append(setEnglish)
    #       seriesBar.setLabelsVisible(True)     #显示每段的标签
    #       seriesBar.setLabelsFormat("@value")
    #       seriesBar.setLabelsPosition(QAbstractBarSeries.LabelsCenter)
    #       #  LabelsCenter,LabelsInsideEnd,LabelsInsideBase,LabelsOutsideEnd
    #       seriesBar.hovered.connect(self.do_barSeries_Hovered) #hovered信号
    #       seriesBar.clicked.connect(self.do_barSeries_Clicked) #clicked信号
    #       chart.addSeries(seriesBar)
    #
    #       axisStud =QBarCategoryAxis()  #类别坐标轴
    #       axisStud.append(nameList)
    #       axisStud.setRange(nameList[0], nameList[stud_Count-1])
    #
    #       axisValue =QValueAxis()    #数值坐标轴
    #       axisValue.setRange(0, 300)
    #       axisValue.setTitleText("总分")
    #       axisValue.setTickCount(6)
    #       axisValue.applyNiceNumbers()
    #
    #       if isVertical:
    #          chart.setAxisX(axisStud, seriesBar)
    #          chart.setAxisY(axisValue, seriesBar)
    #       else:
    #          chart.setAxisY(axisStud, seriesBar)
    #          chart.setAxisX(axisValue, seriesBar)
    #
    #       for marker in chart.legend().markers():  #QLegendMarker类型列表
    #          marker.clicked.connect(self.do_LegendMarkerClicked)
    #
    #
    # ##===========page 3. 百分比柱状图=============
    #    @pyqtSlot()   ##3.1  绘制 PercentBar
    #    def on_btnPercentBar_clicked(self):
    #       self.draw_percentBar()
    #
    #    @pyqtSlot()   ##3.2  绘制 水平PercentBar
    #    def on_btnPercentBarH_clicked(self):
    #       self.draw_percentBar(False)
    #
    #    def draw_percentBar(self,isVertical=True):
    #       chart =self.ui.chartViewPercentBar.chart()
    #       chart.removeAllSeries()
    #       chart.removeAxis(chart.axisX())
    #       chart.removeAxis(chart.axisY())
    #       chart.legend().setAlignment(Qt.AlignRight)   #AlignBottom,AlignTop
    #       if isVertical:
    #          chart.setTitle("PercentBar 演示")
    #       else:
    #          chart.setTitle(" Horizontal PercentBar 演示")
    #
    # ##创建数据集
    #       scoreBarSets=[]   #QBarSet对象列表
    #       sectionCount=5    #5个分数段，分数段是数据集
    #       for i in range(sectionCount):
    #          item=self.ui.treeWidget.topLevelItem(i)
    #          barSet=QBarSet(item.text(0))  #一个分数段
    #          scoreBarSets.append(barSet)   #QBarSet对象列表
    #
    #       categories=["数学","语文","英语"]
    #       courseCount=3   #3门课程
    #       for i in range(sectionCount):   #5个分数段，
    #          item=self.ui.treeWidget.topLevelItem(i)   #treeWidget第i行
    #          barSet=scoreBarSets[i]   #某个分数段的 QBarSet
    #          for j in range(courseCount):   #课程是category
    #             barSet.append(float(item.text(j+1)))
    # ##创建序列
    #       if isVertical:
    #          seriesBar = QPercentBarSeries() #序列
    #       else:
    #          seriesBar = QHorizontalPercentBarSeries() #序列
    #       seriesBar.append(scoreBarSets)      #添加一个QBarSet对象列表
    #       seriesBar.setLabelsVisible(True)    #显示百分比
    #       seriesBar.hovered.connect(self.do_barSeries_Hovered) #hovered信号
    #       seriesBar.clicked.connect(self.do_barSeries_Clicked) #clicked信号
    #       chart.addSeries(seriesBar)
    # ##创建坐标轴
    #       axisSection =  QBarCategoryAxis()   #分类坐标
    #       axisSection.append(categories)
    #       axisSection.setTitleText("分数段")
    #       axisSection.setRange(categories[0], categories[courseCount-1])
    #
    #       axisValue =  QValueAxis()  #数值坐标
    #       axisValue.setRange(0, 100)
    #       axisValue.setTitleText("累积百分比")
    #       axisValue.setTickCount(6)
    #       axisValue.setLabelFormat("%.0f%")   #标签格式
    #       axisValue.applyNiceNumbers()
    #
    #       if isVertical:
    #          chart.setAxisX(axisSection, seriesBar)
    #          chart.setAxisY(axisValue,   seriesBar)
    #       else:
    #          chart.setAxisY(axisSection, seriesBar)
    #          chart.setAxisX(axisValue,   seriesBar)
    #
    #       for marker in chart.legend().markers():  #QLegendMarker类型列表
    #          marker.clicked.connect(self.do_LegendMarkerClicked)

    ##============page 4. 饼图 =====================
    # @pyqtSlot(int)  ##选择课程
    # def on_comboCourse_currentIndexChanged(self, index):
    #     self.draw_pieChart()
    #
    # @pyqtSlot()  ## 绘制饼图
    # def on_btnDrawPieChart_clicked(self):
    #     self.draw_pieChart()
    #
    # @pyqtSlot(float)  ##设置 holeSize
    # def on_spinHoleSize_valueChanged(self, arg1):
    #     seriesPie = self.ui.chartViewPie.chart().series()[0]
    #     seriesPie.setHoleSize(arg1)
    #
    # @pyqtSlot(float)  ##设置pieSize
    # def on_spinPieSize_valueChanged(self, arg1):
    #     seriesPie = self.ui.chartViewPie.chart().series()[0]
    #     seriesPie.setPieSize(arg1)
    #
    # @pyqtSlot(bool)  ##显示图例checkbox
    # def on_chkBox_PieLegend_clicked(self, checked):
    #     self.ui.chartViewPie.chart().legend().setVisible(checked)
    #
    # def draw_pieChart(self):  ##绘制饼图
    #     chart = self.ui.chartViewPie.chart()  # 获取chart对象
    #     chart.legend().setAlignment(Qt.AlignRight)  # AlignRight,AlignBottom
    #     chart.removeAllSeries()  # 删除所有序列
    #
    #     colNo = 1 + self.ui.comboCourse.currentIndex()  # 课程在treeWidget中的列号
    #
    #     seriesPie = QPieSeries()  # 饼图序列
    #     seriesPie.setHoleSize(self.ui.spinHoleSize.value())  # 饼图中间空心的大小
    #     sec_count = 5  # 分数段个数
    #     seriesPie.setLabelsVisible(True)  # 只影响当前的slices，必须添加完slice之后再设置
    #     for i in range(sec_count):  # 添加分块数据,5个分数段
    #         item = self.ui.treeWidget.topLevelItem(i)
    #         sliceLabel = item.text(0) + "(%s人)" % item.text(colNo)
    #         sliceValue = int(item.text(colNo))
    #         seriesPie.append(sliceLabel, sliceValue)  # 添加一个饼图分块数据,(标签，数值)
    #
    #     seriesPie.setLabelsVisible(True)  # 只影响当前的slices，必须添加完slice之后再设置
    #     seriesPie.hovered.connect(self.do_pieHovered)  # 鼠标落在某个分块上时，此分块弹出
    #     chart.addSeries(seriesPie)
    #     chart.setTitle("Piechart---" + self.ui.comboCourse.currentText())

    ##  =============自定义槽函数===============================
    def do_calcuAverage(self, item):  ##计算平均分
        if (item.column() < self.COL_MATH or item.column() > self.COL_ENGLISH):
            return  # 如果被修改的item不是数学、语文、英语,就退出

        rowNo = item.row()  # 获取数据的行编号
        avg = 0.0
        for i in range(self.COL_MATH, 1 + self.COL_ENGLISH):
            item = self.dataModel.item(rowNo, i)
            avg = avg + float(item.text())
        avg = avg / 3.0  # 计算平均分
        item = self.dataModel.item(rowNo, self.COL_AVERAGE)  # 获取平均分数据的item
        item.setText("%.1f" % avg)  # 更新平均分数据

    def do_pieHovered(self, pieSlice, state):  ##鼠标在饼图上移入移出
        pieSlice.setExploded(state)  # 弹出或缩回，具有动态效果
        if state:  # 显示带百分数的标签
            self.__oldLabel = pieSlice.label()  # 保存原来的Label
            pieSlice.setLabel(self.__oldLabel + ": %.1f%%"
                              % (pieSlice.percentage() * 100))
        else:  # 显示原来的标签
            pieSlice.setLabel(self.__oldLabel)

    def do_barSeries_Hovered(self, status, index, barset):  ##关联hovered信号
        hint = "hovered barSet=" + barset.label()
        if status:
            hint = hint + ", index=%d, value=%.2f" % (index, barset.at(index))
        else:
            hint = ""
        self.ui.statusBar.showMessage(hint)

    def do_barSeries_Clicked(self, index, barset):  ##关联clicked信号
        hint = "clicked barSet=" + barset.label()
        hint = hint + ", count=%d, sum=%.2f" % (barset.count(), barset.sum())
        self.ui.statusBar.showMessage(hint)

    def do_LegendMarkerClicked(self):  ##图例单击
        marker = self.sender()  # QLegendMarker

        marker.series().setVisible(not marker.series().isVisible())
        marker.setVisible(True)
        alpha = 1.0
        if not marker.series().isVisible():
            alpha = 0.5

        brush = marker.labelBrush()  # QBrush
        color = brush.color()  # QColor
        color.setAlphaF(alpha)
        brush.setColor(color)
        marker.setLabelBrush(brush)

        brush = marker.brush()
        color = brush.color()
        color.setAlphaF(alpha)
        brush.setColor(color)
        marker.setBrush(brush)

        pen = marker.pen()  # QPen
        color = pen.color()
        color.setAlphaF(alpha)
        pen.setColor(color)
        marker.setPen(pen)

    def iniCoxRes(self):
        self.ui.scrollArea.setWidgetResizable(True)  # 自动调整内部组件大小
        self.ui.scrollArea.setAlignment(Qt.AlignCenter)
        self.ui.cox_label.setAlignment(Qt.AlignCenter)

    @pyqtSlot()
    def on_fresh_list_clicked(self):
        if not os.path.exists(self.survive_path):
            os.makedirs(self.survive_path)
        fileName, tmp = QFileDialog.getOpenFileName(self, 'Open List', '../test-data', '*.xlsx *.jpg *.bmp')
        if fileName is '':
            return
        train_table = pd.read_excel('../test-data/life_line_max_train.xlsx')
        test_table = pd.read_excel('../test-data/life_line_max_test.xlsx')

        for c, i in enumerate(test_table['histological_type']):
            if not 'Colon' in i:
                test_table = test_table.drop(c, axis=0)

        train_os = train_table[['OS_time', 'OS']]
        test_os = test_table[['OS_time', 'OS']]

        train_feature = train_table.iloc[:, 9:]
        test_feature = test_table.iloc[:, 8:]

        train_T = train_feature['OS_time']
        train_E = train_feature['OS']
        test_T = test_feature['OS_time']
        test_E = test_feature['OS']

        cls_list = []
        for i in train_feature.columns[2:]:
            cph = CoxPHFitter()
            cph.fit(pd.concat([train_os, train_feature[i]], axis=1), duration_col='OS_time', event_col='OS')
            if float(cph.summary['p']) <= 0.5:
                cls_list.append(i)

        select_table = train_table[cls_list]
        select_test_table = test_table[cls_list]
        final_train = pd.concat([train_os, select_table], axis=1)
        final_test = pd.concat([test_os, select_test_table], axis=1)

        for col in final_train.columns:
            if 'BACK' in col:
                final_train = final_train.drop(col, axis=1)

        for col in final_train.columns:
            if 'MHLS' in col:
                final_train = final_train.drop(col, axis=1)

        for col in final_test.columns:
            if 'BACK' in col:
                final_test = final_test.drop(col, axis=1)

        for col in final_test.columns:
            if 'MHLS' in col:
                final_test = final_test.drop(col, axis=1)
        #

        # f, aa = plt.subplots(2, 1)
        cph = CoxPHFitter()
        cph.fit(final_train, duration_col='OS_time', event_col='OS', show_progress=True)
        cph.print_summary()
        # aa[0] = cph.plot()
        # aa[1] = cph.plot()
        cph.plot()
        plt.savefig('../dataprocess/survive_data/cox_train_res.jpg', bbox_inches='tight')
        # plt.show()
        plt.close()

        cph1 = CoxPHFitter()
        cph1.fit(final_test, duration_col='OS_time', event_col='OS', show_progress=True)
        cph1.print_summary()
        cph1.plot()
        plt.savefig('../dataprocess/survive_data/cox_test_res.jpg', bbox_inches='tight')
        # plt.show()
        plt.close()

        # fig, ax = plt.subplots(1, 2)

        train_hr_ratio = cph.predict_partial_hazard(final_train)
        kmf = KaplanMeierFitter()
        train_flag = (train_hr_ratio <= 1)
        ax = plt.subplot(111)
        kmf.fit(train_T[train_flag], event_observed=train_E[train_flag], label="low")
        kmf.plot_survival_function(ax=ax)
        kmf.fit(train_T[~train_flag], event_observed=train_E[~train_flag], label="high")
        kmf.plot_survival_function(ax=ax)
        plt.savefig('../dataprocess/survive_data/km_train_res.jpg', bbox_inches='tight')
        # plt.show()

        plt.close()

        test_hr_ratio = cph.predict_partial_hazard(select_test_table)
        test_flag = (test_hr_ratio <= 0.4)
        ax = plt.subplot(111)
        kmf.fit(test_T[test_flag], event_observed=test_E[test_flag], label="low")
        kmf.plot_survival_function(ax=ax)
        kmf.fit(test_T[~test_flag], event_observed=test_E[~test_flag], label="high")
        kmf.plot_survival_function(ax=ax)
        plt.savefig('../dataprocess/survive_data/km_test_res.jpg', bbox_inches='tight')
        # plt.show()

        plt.close()

    @pyqtSlot()
    def on_cox_retrain_clicked(self):
        if not os.path.exists(self.survive_path):
            return
        self.curPixmap = QPixmap()
        self.curPixmap.load('../dataprocess/survive_data/cox_train_res.jpg')
        H = self.ui.scrollArea.height()  # 得到scrollArea的高度
        realH = self.curPixmap.height()  # 原始图片的实际高度
        self.pixRatio = float(H) / realH  # 当前显示比例，必须转换为浮点数
        pix = self.curPixmap.scaledToHeight(H - 30)  # 图片缩放到指定高度
        self.ui.cox_label.setPixmap(pix)  # 设置Label的PixMap
        print('show')

    @pyqtSlot()
    def on_cox_test_clicked(self):
        if not os.path.exists(self.survive_path):
            return
        self.curPixmap = QPixmap()
        self.curPixmap.load('../dataprocess/survive_data/cox_test_res.jpg')
        H = self.ui.scrollArea.height()  # 得到scrollArea的高度
        realH = self.curPixmap.height()  # 原始图片的实际高度
        self.pixRatio = float(H) / realH  # 当前显示比例，必须转换为浮点数
        pix = self.curPixmap.scaledToHeight(H - 30)  # 图片缩放到指定高度
        self.ui.cox_label.setPixmap(pix)  # 设置Label的PixMap

    @pyqtSlot()
    def on_km_retrain_clicked(self):
        if not os.path.exists(self.survive_path):
            return
        self.curPixmap = QPixmap()
        self.curPixmap.load('../dataprocess/survive_data/km_train_res.jpg')
        H = self.ui.scrollArea.height()  # 得到scrollArea的高度
        realH = self.curPixmap.height()  # 原始图片的实际高度
        self.pixRatio = float(H) / realH  # 当前显示比例，必须转换为浮点数
        pix = self.curPixmap.scaledToHeight(H - 30)  # 图片缩放到指定高度
        self.ui.cox_label.setPixmap(pix)  # 设置Label的PixMap

    @pyqtSlot()
    def on_km_test_clicked(self):
        if not os.path.exists(self.survive_path):
            return
        self.curPixmap = QPixmap()
        self.curPixmap.load('../dataprocess/survive_data/km_test_res.jpg')
        H = self.ui.scrollArea.height()  # 得到scrollArea的高度
        realH = self.curPixmap.height()  # 原始图片的实际高度
        self.pixRatio = float(H) / realH  # 当前显示比例，必须转换为浮点数
        pix = self.curPixmap.scaledToHeight(H - 30)  # 图片缩放到指定高度
        self.ui.cox_label.setPixmap(pix)  # 设置Label的PixMap


##  ============窗体测试程序 ================================
if __name__ == "__main__":  # 用于当前窗体测试
    app = QApplication(sys.argv)  # 创建GUI应用程序
    form = QmyMainWindow(None)  # 创建窗体
    form.show()
    sys.exit(app.exec_())
