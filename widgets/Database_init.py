from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel, QSqlRecord
from slide_viewer.common.qt.myDelegates import QmyComboBoxDelegate


##  ==============数据库部分自定义功能函数============##
def __getFieldNames(self):  ##获取所有字段名称
    emptyRec = self.tabModel.record()  # 获取空记录，只有字段名
    self.fldNum = {}  # 字段名与序号的字典
    for i in range(emptyRec.count()):
        fieldName = emptyRec.fieldName(i)
        self.comboFields.addItem(fieldName)
        self.fldNum.setdefault(fieldName)
        self.fldNum[fieldName] = i
    print(self.fldNum)


def __openTable(self):  ##打开数据表打开的db只是一个格式空的  内容需要自己写
    self.tabModel = QSqlTableModel(self, self.DB)  # 数据模型
    self.tabModel.setTable("employee")  # 设置数据表  在数据库中的单个数据列表名称  设置需要连接的数据表
    self.tabModel.setEditStrategy(QSqlTableModel.OnManualSubmit)  # 数据保存方式，OnManualSubmit , OnRowChange
    self.tabModel.setSort(self.tabModel.fieldIndex("empNo"), Qt.AscendingOrder)  # 排序
    if (self.tabModel.select() == False):  # 查询数据失败
        QMessageBox.critical(self, "错误信息",
                             "打开数据表错误,错误信息\n" + self.tabModel.lastError().text())
        return

    self.__getFieldNames()  # 获取字段名和序号

    ##字段显示名
    self.tabModel.setHeaderData(self.fldNum["empNo"], Qt.Horizontal, "患者编号")
    self.tabModel.setHeaderData(self.fldNum["Name"], Qt.Horizontal, "姓名")
    self.tabModel.setHeaderData(self.fldNum["Gender"], Qt.Horizontal, "性别")
    self.tabModel.setHeaderData(self.fldNum["Birthday"], Qt.Horizontal, "年龄")
    self.tabModel.setHeaderData(self.fldNum["Province"], Qt.Horizontal, "患癌类型")
    self.tabModel.setHeaderData(self.fldNum["Department"], Qt.Horizontal, "癌症亚型")
    self.tabModel.setHeaderData(self.fldNum["Salary"], Qt.Horizontal, "预后分析")

    self.tabModel.setHeaderData(self.fldNum["Memo"], Qt.Horizontal, "备注")  # 这两个字段不在tableView中显示 任意长度的普通文本
    self.tabModel.setHeaderData(self.fldNum["Photo"], Qt.Horizontal, "照片")

    ##创建界面组件与数据模型的字段之间的数据映射
    self.mapper = QDataWidgetMapper()
    self.mapper.setModel(self.tabModel)  # 设置数据模型
    self.mapper.setSubmitPolicy(QDataWidgetMapper.AutoSubmit)

    ##界面组件与tabModel的具体字段之间的联系     两个框架数据之间的连接方式
    self.mapper.addMapping(self.dbSpinEmpNo, self.fldNum["empNo"])
    self.mapper.addMapping(self.dbEditName, self.fldNum["Name"])
    self.mapper.addMapping(self.dbComboSex, self.fldNum["Gender"])
    self.mapper.addMapping(self.dbEditBirth, self.fldNum["Birthday"])
    self.mapper.addMapping(self.dbComboProvince, self.fldNum["Province"])
    self.mapper.addMapping(self.dbComboDep, self.fldNum["Department"])
    self.mapper.addMapping(self.dbSpinSalary, self.fldNum["Salary"])
    self.mapper.addMapping(self.dbEditMemo, self.fldNum["Memo"])
    self.mapper.toFirst()  # 移动到首记录

    self.selModel = QItemSelectionModel(self.tabModel)  # 选择模型
    self.selModel.currentChanged.connect(self.do_currentChanged)  # 当前项变化时触发
    self.selModel.currentRowChanged.connect(self.do_currentRowChanged)  # 选择行变化时

    self.tableView.setModel(self.tabModel)  # 设置数据模型
    self.tableView.setSelectionModel(self.selModel)  # 设置选择模型

    self.tableView.setColumnHidden(self.fldNum["Memo"], True)  # 隐藏列
    self.tableView.setColumnHidden(self.fldNum["Photo"], True)  # 隐藏列

    ##tableView上为“性别”和“部门”两个字段设置自定义代理组件
    strList = ("男", "女")
    self.__delegateSex = QmyComboBoxDelegate()
    self.__delegateSex.setItems(strList, False)
    self.tableView.setItemDelegateForColumn(self.fldNum["Gender"], self.__delegateSex)  # Combbox选择型

    strList = ("销售部", "技术部", "生产部", "行政部")
    self.__delegateDepart = QmyComboBoxDelegate()
    self.__delegateDepart.setItems(strList, True)
    self.tableView.setItemDelegateForColumn(self.fldNum["Department"], self.__delegateDepart)

    ##更新actions和界面组件的使能状态
    self.act_database.setEnabled(False)

    self.actRecAppend.setEnabled(True)
    self.actRecInsert.setEnabled(True)
    self.actRecDelete.setEnabled(True)
    self.actScan.setEnabled(True)

    self.groupBoxSort.setEnabled(True)
    self.groupBoxFilter.setEnabled(True)


@pyqtSlot()  ##选择数据库，打开数据表
def on_act_database_triggered(self):
    dbFilename, flt = QFileDialog.getOpenFileName(self, "选择数据库文件", "",
                                                  "SQL Lite数据库(*.db *.db3)")
    if (dbFilename == ''):
        return

    # 打开数据库
    self.DB = QSqlDatabase.addDatabase("QSQLITE")  # 添加 SQLITE数据库驱动
    self.DB.setDatabaseName(dbFilename)  # 设置数据库名称
    # self.DB.setHostName('55')
    # self.DB.setUserName('wch')
    # self.DB.setPassword('123')
    if self.DB.open():  # 打开数据库
        self.__openTable()  # 打开数据表
    else:
        QMessageBox.warning(self, "错误", "打开数据库失败")


@pyqtSlot()  ##保存修改
def on_actSubmit_triggered(self):
    res = self.tabModel.submitAll()
    if (res == False):
        QMessageBox.information(self, "消息",
                                "数据保存错误,错误信息\n" + self.tabModel.lastError().text())
    else:
        self.actSubmit.setEnabled(False)
        self.actRevert.setEnabled(False)


@pyqtSlot()  ##取消修改
def on_actRevert_triggered(self):
    self.tabModel.revertAll()
    self.actSubmit.setEnabled(False)
    self.actRevert.setEnabled(False)


@pyqtSlot()  ##添加记录
def on_actRecAppend_triggered(self):
    self.tabModel.insertRow(self.tabModel.rowCount(), QModelIndex())  # 在末尾添加一个记录

    curIndex = self.tabModel.index(self.tabModel.rowCount() - 1, 1)  # 创建最后一行的ModelIndex
    self.selModel.clearSelection()  # 清空选择项
    self.selModel.setCurrentIndex(curIndex, QItemSelectionModel.Select)  # 设置刚插入的行为当前选择行

    currow = curIndex.row()  # 获得当前行
    self.tabModel.setData(self.tabModel.index(currow, self.fldNum["empNo"]),
                          2000 + self.tabModel.rowCount())  # 自动生成编号
    self.tabModel.setData(self.tabModel.index(currow, self.fldNum["Gender"]), "男")  ##填入默认值 每个字占四个字节


@pyqtSlot()  ##插入记录
def on_actRecInsert_triggered(self):
    curIndex = self.ui.tableView.currentIndex()  # QModelIndex
    self.tabModel.insertRow(curIndex.row(), QModelIndex())
    self.selModel.clearSelection()  # 清除已有选择
    self.selModel.setCurrentIndex(curIndex, QItemSelectionModel.Select)


@pyqtSlot()  ##删除记录
def on_actRecDelete_triggered(self):
    curIndex = self.selModel.currentIndex()  # 获取当前选择单元格的模型索引
    self.tabModel.removeRow(curIndex.row())  # 删除当前行


@pyqtSlot()  ##清除照片
def on_actPhotoClear_triggered(self):
    curRecNo = self.selModel.currentIndex().row()
    curRec = self.tabModel.record(curRecNo)  # 获取当前记录,QSqlRecord
    curRec.setNull("Photo")  # 设置为空值
    self.tabModel.setRecord(curRecNo, curRec)
    self.dbLabPhoto.clear()  # 清除界面上的图片显示


@pyqtSlot()  ##设置照片
def on_actPhoto_triggered(self):
    fileName, filt = QFileDialog.getOpenFileName(self, "选择图片文件", "", "照片(*.jpg)")
    if (fileName == ''):
        return

    file = QFile(fileName)  # fileName为图片文件名
    file.open(QIODevice.ReadOnly)  # 设置为只读模式
    try:
        data = file.readAll()  # QByteArray字节列表
    finally:
        file.close()

    curRecNo = self.selModel.currentIndex().row()
    curRec = self.tabModel.record(curRecNo)  # 获取当前记录QSqlRecord
    curRec.setValue("Photo", data)  # 设置字段数据
    self.tabModel.setRecord(curRecNo, curRec)

    pic = QPixmap()
    pic.loadFromData(data)
    # W = self.dbLabPhoto.width()
    W = 130
    self.dbLabPhoto.setPixmap(pic.scaledToWidth(W))  # 在界面上显示也需要QPixmap


@pyqtSlot()  ##涨工资，遍历数据表所有记录   后面需要更改
def on_actScan_triggered(self):
    if (self.tabModel.rowCount() == 0):
        return

    for i in range(self.tabModel.rowCount()):
        aRec = self.tabModel.record(i)  # 获取当前记录
        ##         salary=aRec.value("Salary").toFloat()      #错误，无需再使用toFloat()函数
        salary = aRec.value("Salary")
        salary = salary * 1.1
        aRec.setValue("Salary", salary)
        self.tabModel.setRecord(i, aRec)

    if (self.tabModel.submitAll()):
        QMessageBox.information(self, "消息", "涨工资计算完毕")


@pyqtSlot(int)  ##排序字段变化
def on_comboFields_currentIndexChanged(self, index):
    if self.radioBtnAscend.isChecked():
        self.tabModel.setSort(index, Qt.AscendingOrder)
    else:
        self.tabModel.setSort(index, Qt.DescendingOrder)
    self.tabModel.select()


@pyqtSlot()  ##升序
def on_radioBtnAscend_clicked(self):
    self.tabModel.setSort(self.comboFields.currentIndex(), Qt.AscendingOrder)
    self.tabModel.select()


@pyqtSlot()  ##降序
def on_radioBtnDescend_clicked(self):
    self.tabModel.setSort(self.comboFields.currentIndex(), Qt.DescendingOrder)
    self.tabModel.select()


@pyqtSlot()  ##过滤，男
def on_radioBtnMan_clicked(self):
    self.tabModel.setFilter("Gender='男'")


@pyqtSlot()  ##数据过滤，女
def on_radioBtnWoman_clicked(self):
    self.tabModel.setFilter("Gender='女' ")


@pyqtSlot()  ##取消数据过滤
def on_radioBtnBoth_clicked(self):
    self.tabModel.setFilter("")


def do_currentChanged(self, current, previous):  ##更新actPost和actCancel 的状态
    self.actSubmit.setEnabled(self.tabModel.isDirty())  # 有未保存修改时可用
    self.actRevert.setEnabled(self.tabModel.isDirty())


def do_currentRowChanged(self, current, previous):  # 行切换时的状态控制
    self.actRecDelete.setEnabled(current.isValid())
    self.actPhoto.setEnabled(current.isValid())
    self.actPhotoClear.setEnabled(current.isValid())

    if (current.isValid() == False):
        self.dbLabPhoto.clear()  # 清除图片显示
        return

    self.mapper.setCurrentIndex(current.row())  # 更新数据映射的行号
    curRec = self.tabModel.record(current.row())  # 获取当前记录,QSqlRecord类型

    if (curRec.isNull("Photo")):  # 图片字段内容为空
        self.dbLabPhoto.clear()
    else:
        ##         data=bytearray(curRec.value("Photo"))   #可以工作
        data = curRec.value("Photo")  # 也可以工作
        pic = QPixmap()
        pic.loadFromData(data)
        # W = self.dbLabPhoto.size().width()
        W = 130
        self.dbLabPhoto.setPixmap(pic.scaledToWidth(W))