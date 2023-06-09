PathPlatform Application
=============================

This repository contains a code for QT for Python based application.

##开发日志

###7.17
- 主要任务：  
   1. 按键响应鼠标选择tiles进一步筛选预处理区域
   2. 为`slidehelper`添加返回参数
   3. `MouseEvent`添加传参
   4. 右键菜单`Mainwindow`中直接指定`scrollarea`避免边界判断错误
   5. 为分割进程创建结束返回弹窗
   6. 鼠标响应时，双参数选择问题
   7. 鼠标相应下放到参数选择后面出现问题无法案件松开取消操作  改用·state·方案进行尝试  
      发现是return使用的问题没有顺序执行下去
   8. 没有`mousemove` 事件无法画框  用于为`Rubberband`指定框选区域和位置
   9. middlebutton程序段无意义

###7.21
-  主要任务：  
    -  做`index`索引行列更改pos neg  逻辑为：
         - 对遮盖后的污染pos区域算出框选出的区域col row范围
         - 通过保存的`pos_index`转化成list遍历搜索 行列并将其对应删除
         - 在算法处理的时候只需要根据list对应读取就行了
    - 框选算法逻辑:
         - 左上右下除去当前界面`gridsize`大小，取整向前计算   
         - 图片保存时去除行列的空格直接检索  
         - 注意grid画框的层级问题  
         - 图像移动后会发生坐标错乱问题  不做拖动可以避免坐标错乱
         - 注意画框的随机性  需要进行不可撤销性的设置 
         - 将所有选中的行数和列数全部保存后写成一个新的数组进行查找index对应的行返回行的序号统计完成后一起删除
         - 重写底层涂抹原理，点选涂抹
         - 鼠标相应在高倍放大切片下 底层程序存在鼠标位置漂移现象
         - 图片信息和场景信息放到目录树里

###7.23
-  主要任务：  
    -  点选pos和框选pos的函数要区分开
    -  可以在框选循环里加入`annotation`循环基本不会影响运行速度
    -  txt文件采用w--write擦除   a--writelines写入不擦除
    -  单选的并列显示问题搞定  取消`clearlevel`的操作
    -  单块点选需要取整`gridsize`保证与分割结果对应
    -  解决txt复用重新写入和 继续写入  全文擦除的选择问题

###7.24
-  主要任务：  
    -  分割任务中用线程池优化  加速
    -  "\t" 对齐空格
    -  pool 传参数量问题只可以传参一个
    -  传单个参数后剩下的参数重新计算
    -  线程数大概在35左右吃满cpu  多线程加速感知不明显
    -  线程数最好与cpu契合 i5 9500 6线程吃满   一般`pool(3)`保证流畅高效
    -  因多线程输出需要重写背景筛选
    -  多线程运行时会在python主进程里空置两个线程   有一个资源线程命名为窗口界面名
       运行时最好流出界面线程保证窗口不卡
    -  后面尝试把分割的参数封装到helper里面

###7.28
-  主要任务：  
    -  图景移动通过 `setFlag-moveable`实现的需要具体评估移动的坐标对应问题
    -  坐标错乱原因可能是移动后的图层变量占用了网格操作的变量导致操作不准确
    -  图景moveable设置后移到鼠标判断部分进行整合
    -  在`buildbackground`里更改多线程的alpha紊乱问题
       双`for`循环嵌套筛选`coloralpha`数组
    -  解决多线程传参问题
    -  多线程操作时为其传入slidename参数
    -  多线程处理大的切片背景信息进行循环嵌套获得alpha数组时会浪费较多的时间  
       存在优化空间
    -  优化分割效率
    -  缩略图点击显示映射现在会刷新图元 需要将其下放
    -  `Location`小块tiles定位
    
###7.29
-  主要任务：  
    -  通过偏函数的方式成功实现多线程固定一个变量值的多变量传参
    -  偏函数不可封装不定值变量  ex `slide_dz`
    -  完成多线程优化
    -  通过open index 'w' 'a'切换解决了多线程时重复写入问题
    -  在移动坐标混乱场景中要让 00 起始位置随鼠标一起移动
    -  框选功能需要精细化为对应底层小块
    -  最好取消点击移动图元功能，会造成坐标紊乱 ，为了和其它功能保持稳定必须同时移动scene会造成紊乱，
       scene和view 需要趋同，从添加网格的坐标角度再看一下可行性 
       以pm坐标进行绘制
    -  变量`self.slide_graphics.scenePos()` 为图源相对 `(0,0)` 位置移动距离
       对其进行绘制时的传参操作
    -  更新后存在区域边缘刷新涂抹问题 `init_grid_levels`存在问题需要传参
    -  更新后放缩仍有问题 色块显示存在反应时间
    -  背景滤除时多线程加速

    
###8.3
-  主要任务：  
    -  不进行拖动移动时不会出现坐标紊乱的情况
    -  缩略图和定位刷新无法改变
    -  整合背景滤除和显示网格
    -  多进程操作中：join([timeout]) 如果可选参数timeout是 None（默认值），则该方法将阻塞，
       直到调用 join() 方法的进程终止。
       简单说哪个子进程调用了join方法，主进程就要等该子进程执行完后才能继续向下执行
    -  QThread 不好用无法在根本上解决多进程传参问题
    -  多进程的结束标志获取是个大问题否则无法完成弹窗 需要通过timer的方式在运行过程中随时监控
       `active`的bool返回值
    -  不可以把多进程封装在在同一类之下
    -  采用`pipe()`管道通信的方式也无法解决进程弹窗问题
    -  对pool线程池的分割操作出现了降频现象无法高效使用 cpu      
       不使用 `join,colse` 不会发生  玄学问题  降频可能和室温有关
    -  背景筛选弹窗弹出后会导致网格绘制出现暂时错乱

###8.7
-  主要任务：  
    -  鼠标右键菜单放到主窗口下方便功能函数传参操控
    -  点按对应区域显示原色去掉选色框
    -  概率 * 255 某个颜色的权值 进行部分色块区域的重绘


###8.18
-  主要任务：  
    -  将每个不同的alpha值对应的区域各自整理成集合去绘制 `gridgraphicsitem`
       ```python
       for color_alpha, grid_rect_0_level in zip(color_alphas, grid_rects_0_level):
            self.color_alpha_rects_0_level.setdefault(color_alpha, []).append(
               grid_rect_0_level )
       for color_alpha, rects in self.color_alpha_rects_0_level.items():
            color = QColor(*self.base_color_rgb, color_alpha)
    -  为for循环加阻塞可解决降频问题  原因是线程池推多线程时的死循环问题导致cpu占用无法释放
    -  先做九选一全局同色显示 背景滤掉后标黑
    -  随机生成的患癌率因为是九选一取最大所以要足够高
    -  判断概率为255直接背景标黑
    -  painter 在鼠标移动时会随时刷新避免在其中写入过多循环体
    -  rgb转HSV实现同类别同色域不同深度的处理
    -  在涂抹压缩时采用九类分类标签去做处理简化画图流程    
    -  涂抹时rect循环体内已经有了循环变量是进行的单个赋值不需要将coloralpha执行循环列表化
    -  `slidegroup` # self.leveled_graphics_selection.clear_level(level)  ##控制并列选框
    -  设置为每一类同色同清晰度，第一版九分类同色集合显示
    -  生成的筛选alpha 继续套用for循环绘图位置出现问题
    -  HCV显色时0对应黑色
    
    
###9.1 
-  主要任务：  
    -  将tiles进行255背景复制 患癌概率高的进行255取反保持hsv中的深色
    -  将保存信息增加九个后因为读取到数组也需要九个效率会降低 
    -  注意保存信息时的空格问题  字符形式的空格过多会导致程序报错
    -  `gridratio`原始切片的最高放大倍数与20倍之比 
    -  使用O(n2)复杂度实现部分特征多选的数组运算  运算保持了高效
    -  实现了单类特征不透明alpha = 255 的选择单色域显色处理
       根据患癌概率值的不同进行zip压缩分组对应同一色域不同颜色
    -  实现了全特征选择的处理   因使用的保存数组不同   显示效果不同   后期进行合并   去掉随机生成的
       类别和alpha 数组 
    -  重写透明网格绘制部分
    -  不对背景绘制  重新引入透明度设置
    -  添加患癌概率阈值筛选
    -  进行绘制的滑块筛选的时候如果刷新绘制会导致无法在全部层级实现绘制
    -  通过按键状态的bool值进行刷新绘制区域
    -  重写滑动条部分
    -  患癌概率阈值筛选的两个思路: 1. 图元分割时引入阈值筛选
                               2. 绘制数组创建 `level-builder` 引入阈值判断
    
    
###9.14
-  主要任务：  
    -  在 `groupbox` 中进行enable设置可以激活里面的滑动条使用
    -  重绘图元时可以改变 `update_grid_visibility`的bool赋值实现刷新
    -  后期对于大图的刷新绘制考虑多线程加速
    -  对于绘制的效率问题  双参数for循环次数是影响运行效率的主要因素
       单线程吃到20%左右  
    -  弹窗选类时引入患癌概率筛选而不是透明度
    -  绘制的处理速度是个大问题
    -  绘制效率与 `tiles`个数负相关 与图源内存占用没有直接关系
    -  ``POOL`` 直接推 `alpharects` 会将所有数据推入需要重新找入池变量 完成参数传递
    -  为保证深度网络resnet的输入要求224*224 要在分割时对边缘数据进行筛除 去除在边缘的positive部分
       尝试row col减一的方式  会导致绘制混乱
       在`deepzoom_tile_pool`中进行分割信息的保存更改判断   改为交集 `and` 判断为 True 保存信息完美解决
    -  由于深度网络处理后打乱了原来的顺序  rects 和 row col的对应顺序   需要在bulid数组的时候重新排一下序
       采用三层for循环嵌套可以实现完美排序 但会导致运行效率降低 
    -  进度条加标值 用了自定义进度条的的方式实现   将进度条位置置于groupboxlayout中
    -  深度resnet处理时cancertype需要+1不要用错成数组顺序
    -  resnet多进程下导包出现问题    
       可能是分割任务里额外定义了一个类导致
       进行测试时因导包问题导致测试运行时过量占用内存  导致闪退
       `torch`,`torchvision` 单独使用没有问题在pyqt界面里嵌套导包是出现问题
    -  绘制过程多线程加速:   #如果要进行数组间一一对应则无法用多线程推线程池的方法加速



###9.28
-  主要任务： 
    -  为了进行绘制尝试在加载大图时对大图进行分割
    -  对于单张切片 癌组织往往只有一类大范围区域  其余癌旁区域其他亚型也基本只有一种最显著  其他的亚型概率往往很低
    -  resnet无法导包问题在更新了torch torchvision 以及改变了 `torch.device`指定方式之后解决了问题
    -  在选择多类时 `level builders`中时间复杂度为 `O(n^2)`的 for循环是为了部分排序
    -  尝试使用resnet进行单张处理无效  改变思路将RECTS变为二维数组进行 col row对应  
       先升维后降维 大大缩短运行时间   完美绘制  
       思路 将 `RECTS, ALPHA ，row，col， type，rate  `一一对应不需要原来 `O(n^3)`的算法大大缩短运行时间
       ```python
       area = np.array(list(rects)).reshape(rows, cols, 4)
       fix_area = []
       for annotation in annotations:
          annotation = annotation.strip().split(' ')
          row = int(annotation[3])
          col = int(annotation[2])
          fix_area.append(area[row][col])
          background_choose = int(annotation[1])
          color_alphas.append(background_choose)
          cancer_type.append(int(annotation[4]))
    -  注意col row 在 `reshape`以及数组添加元素时要与绘制时的顺序保持一致
    -  时间复杂度为一比原先的还要快 不再需要多线程绘制  开始考虑神经网络运算的加速 
    -  癌类图标颜色区分 注释可移动tiles点对鼠标显示对应的患癌概率
    -  做体素级的overleap 进行概率精细化   
    -  heatmaplegend  `COPYNUM_LABELS` 为图例区域
    -  大部分绘图都是使用的matplotlib包封装好的图例
    -  重写一个绘制小块 
    
###9.29
-  主要任务：    
    -  进一步精简程序去除冗余段
    -  进行数据倍增时要更改rects 和 alpha 数组的构建
       - 数据倍增主要进行cuttiles保存时进行处理
       - 当前的分割任务行列数是直接计算好的  暂存在dzlevel
         体素级的分割可能迟滞
       ``` python
        slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)
       ``` 
       - overlap 参数无效 为边线所占体素数
       - 进行更改svs底层大小进行体素平移for循环的方法 尝试进行数据扩增 需要兼顾多进程
       - 采用112打散重组的方式进行数据扩增在分割的时候时间复杂度较高  无法使用多线程 不采纳这个思路
    
###10.14
-  主要任务：    
    -  重写数据预处理部分 进行污染物的背景筛选滤除
    -  pool线程池
       - python 进程池pool简单使用
         - apply()
           函数原型：apply(func[, args=()[, kwds={}]]) 
           该函数用于传递不定参数，同python中的apply函数一致，主进程会被阻塞直到函数执行结束
         - apply_async()
           函数原型：apply_async(func[, args=()[, kwds={}[, callback=None]]])
　　　　    与apply用法一致，但它是非阻塞的且支持结果返回后进行回调。
         - map()
           函数原型：map(func, iterable[, chunksize=None])
　　　　    Pool类中的map方法，与内置的map函数用法行为基本一致，它会使进程阻塞直到结果返回。 
　　　　    注意：虽然第二个参数是一个迭代器，但在实际使用中，必须在整个队列都就绪后，程序才会运行子进程。
         - map_async()
　　　　    函数原型：map_async(func, iterable[, chunksize[, callback]])
　　　　    与map用法一致，但是它是非阻塞的。其有关事项见apply_async。
         - close()
　　　　    关闭进程池（pool），使其不在接受新的任务。
         - terminal()
　　    　　结束工作进程，不在处理未处理的任务。
         - join()
　　　    　主进程阻塞等待子进程的退出， join方法要在close或terminate之后使用。
    - 源代码结构复杂 冗余程序段过多
    - 小型连通域去除算法
    - 连通域判定算法    二值图像分析 四方向邻接 八方向邻接
      对像素点进行二值打标
      - Two-Pass 算法 先行后列
      - Seed-Filling 算法 种子填充生长

###10.23
-  主要任务：    
    -  传参元数据指定函数 `get_training_slide_path`
    -  预处理任务添加一个多进程
    -  将预处理任务滞后与多进程同时执行会导致明显卡顿
    -  前期预处理连接完毕进行显示方式模块的设计 html格式
    -  win下的  \ 默认为 `\\`  用 `PATH.replace('\\','/')`进行替换 完成win系统下的读取
    
###11.12
-  主要任务：    
    -  整合不同的染色归一化方式  添加按钮选项 
    -  完成染色归一化
    -  安装spambin包之后可以正常运行spam
    -  加入概率的状态栏   解决鼠标点击移动更新问题 尝试通过同步移动scene的方式进行****
    -  新的class传参方式
    -  进行整合打包规避递归深度过深需要在 spec文件进行sys生命
    -  更改56*56size 需要更改绘制时的size和去掉原网络无效输出参数 features
    -  打包报错 [14468] WARNING: file already exists but should not: C:\Users\wch\AppData\Local\Temp\_MEI144682\torch\_C.cp37-win_amd64.pyd 
    -  为每次不同尺寸的分割结果创建不同目录的保存文件夹 防止深度网络处理数据格式尺寸错误
    -  框选 点选切片暂时使用224 不做自动更改 
    -  暂时关闭框选时的list修改更新功能
    -  弱监督二分类网络配置 MIL_2_class  224尺寸   无需染色归一化
       ``` python
       from torchvision import models
            model = models.resnet18()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            [0.001356384134851396, 0.9986435770988464]  输出变量个数
       ``` 
###11.30
-  主要任务：   
    - 去除多进程 深度网络  进行染色归一化可以打包不会出现递归深度溢出的情况  
    - 可以运行需要较长时间打开   染色归一化中的opencv调用出现问题   去掉归一化重新尝试
    - 染色归一化去掉后并不会使生成的.exe 变小 大小只与import 有关导入了都会打包
    -  multiprocessing.freeze_support()  /main.py
       加入这句话  解决打包多线程多进程问题 
    - 归一化打包出现问题是因为参考图片指定地址方式出现了问题 导致没有读入图片
    - os.path.abspath('colornorm1.jpg')  dist文件夹下放入参考图片  绝对路径读取
    - 简化打包后可以运行 体积较大需要较长时间打开功能没有问题   忽略警告  需要优化torch
    - 重做list svs在net存储中生成 
    - 用-D命令打包可加快打开速度
    
###12.17
-  主要任务：   
    - 数据库功能置入总体目录   整合数据库功能 独立重写将数据库部分功能脱离主窗口模块化
    - 数据库目录检索页面的初始内容是在设计ui的时候设置的 
    - slide变量不可以作为偏函数参数传参
    - 鼠标浮动对应状态栏刷新显示概率
    - 打包递归深度溢出的情况
    - 数据库信息填全了保存才不会报错 submit返回true
    - 不同需求下的图注跳转-
    - 通过画背景按键进行概率值返回
    - 绘制时的数组返回并不会导致效率降低  几乎无影响  可以进行状态栏实时对应
    - 状态栏需要向下对应
    - 限制窗口页面最小缩放的原因可能是状态栏的最小宽度参数指定
    -   ```python
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_memory_usage) 
        #通过设置计数器刷新内存占用显示
    - 选择框工具栏永远都是获取最底层尺寸   还要考虑分割时的层级倍数选择问题
    - 需要通过for对应多个数组并进行数据暂存
    - 添加一个已经完成深度处理的数据标志 param 通过状态控制状态栏刷新
    - self.slide_view_params.grid_visible,self.slide_graphics.slide_view_params.grid_visible
      两个参数等价 可以控制状态栏刷新   rate数组不会因状态改变而改变 传参后一直有值
      TRUE 时传参计算区域与概率对应
    - 不再采用向上传参的方法徒增功耗
    - 后期需要对没有完全分割完的图像新曾文件夹获取判断方式
    - 向helper里传入值包括行列号 
    - 改变overlap 参数会导致没有图像分割结果把保存 可能是行列数获取出了问题
    - 完善整个多模块选择选项 通过tiles数量判断是否执行分割

###12.22
-  主要任务：  
    - slideviewparams 里的参数可以随着传参刷新 增加一个rate = None 的执行判断体
    - ```python
      if mouse_rects_dict:
          print(key)  
      dict.get(key)   对于不存在的key不会报错keyerror
      字典非空的判断方式
      通过字典查询避免排序降低时间复杂度

    - 透明度在全局的调整   和   癌类筛选的鼠标概率对应  
      不同尺寸下参数的不同操作 gridsize重新传参
      内存速率不够用大文件卡顿  
    - 鼠标状态栏刷新是全类别选择
    - 对于滑动条通过状态判断  让其在不进行癌类筛选时进行全种类的显示
    - 将两种mask 合并  通过指定selecttype参数为1-9
    - 大文件的滑动条调整会闪退   一方面是内存占用剧增  另一方面是cpu单进程运算无法调动全局资源
      通过对生成的mask进行调整参数不再重复生成 或进行多进程绘制尝试解决
      调整内存管理  优化mask生成方式
    - 微卫星稳定性整合  意义引入
    - 每经过一次绘制刷新网格线深度都会加深
    - QPixmapCache.clear()清空加载多张图片时的缓冲区 效用不大  
    - 重新规划预处理部分
    - 封装好的前背景分割功能放到分割进程里面独立开来
      打包这部分的时候会导致闪退   直接放弃这个功能  
      为防止数据重复存储读不到位置  每次处理前将data文件夹重建  前背景分割的tiles功能可以暂时不置入
      print语句需要精简打印内容太多了徒增功耗
    - 重写数据库目录树的结构  更改切片blob的动态存储
    
###12.28
-  主要任务：
    - 微卫星稳定性引入
    - 数据库构建
    - 比划映射清除 需要在背景预处理部分嵌入 但背景判断是在tiles层面做的处理
      原本的224神经网络能进行一定程度上的背景清除 但做不到特别精细化
      112 也可以基本滤除
      改用56 基本可以做到滤除
      - 计算大文件时内存爆了   
    - 打包好的文件 鼠标点击和滚轮有无效输出print   应该和label update 有关  改了还是没用
      目前没有解决方式 可能是save和restore没有对应好
    - gpu分割加速  openslide底层问题   分割时的效率问题很大程度上和硬盘速率有关
    - 染色归一化 和背景滤除不会占用过多处理资源
    - 切片处理部分不再具备优化空间
    

###2021.1.4
-  主要任务：
    - 重改背景区分阈值
    - 新数据之间差异较大 背景较难分辨 设置动态阈值
    - 扩增pycharm内存占用
    - 九类对应名称、
    
###2021.3.23
-  主要任务： 
    - 图片读取放在TileGraphicsItem类中进行实现  借用Painter实现
    - 找寻不能显示普通图片的原因  在上面层面上更改不可行深度太深了
    
    - 对于ROI 保存  可以通过向参数保存函数传参   按键后readregion保存区域  直接定位到最高放大倍数层级
      按键保存时功能触发了两次
    - QPushButton按钮有两个信号函数clicked ，一个带参数clicked(bool)，一个不带参数clicked()，
      当不对槽函数的参数进行限定时，每clicked一次，两个信号都触发，因而也调用两次槽函数。
      加装饰器完美解决
    - 进行动态笔划时采用宽度为一小方格方式存在跟随拖影问题   动态刷新速率不够
    - 通过定义clear——level 清楚松开按键时的比划
      print(pm.x() * level_downsample, pm.y() * level_downsample)  ##实时位置
      pt 为scene坐标系  pm 为item坐标系
    - 画圈自动连接实现  记录一个按下初值画出一个矩形区域  终点松开时落入这个区域自动连接  
      通过计算绝对距离实现画圈的封闭   阈值设置为20000（含有相对位置乘方）   140左右的距离
      在鼠标松开的条件判断里  对封闭区间进行绘制  取最贴合封闭区域间的矩形区域进行roi获取
      既在绘制过程中的四个极值
    - 实现一个橡皮擦的功能  对区域进行边缘校正
    - 显示区域存在滑动加载的区域范围  拖动滑动条加载图元扩大内存占用
    - resnet50  效果更好
    - 开通多进程进行多个网络不通结果的对比呈现
    - 针对传统打包方式的前背景筛除打包后因matplotlib包的打包报错
    - 目前确认是Pyside2库 和 matplotlib库打包在一起导致了程序无法正常运行起来。
      解决方法有2个方法：

      在打包时候将Pyside2库给排除，不进行打包。
      直接pip uninstall pyside2
      
      
c:\anaconda3\lib\site-packages\PyInstaller\loader\pyimod03_importers.py:623: MatplotlibDeprecationWarning:
The MATPLOTLIBDATA environment variable was deprecated in Matplotlib 3.1 and will be removed in 3.3.
  exec(bytecode, module.__dict__)

pip uninstall matplotlib  # 卸载原来的版本
pip install matplotlib==3.1.1 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com  # 安装3.1版本



在本地禁用matplotlib（matplotlib/__init__.py:625）中的弃用警告

找到这个文件的625行

禁用警告

全部注释掉就行了。 
  
源程序中有对于matpoltlib的使用打包无问题   背景过滤部分的代码调包使用有问题
目标函数 tiles.np_hsv_hue_histogram  np_histogram  进行修改   测试发现之前的调用用不到这个功能'
除这些之外还有其他的问题  
打包filters就闪退     

###2021.4.23
-  主要任务： 
    - 1.slide打包没问题  skimage的问题
    - 2.包中缺少dask包dask.yaml  手动载入可以打开
    - 3.对于文件夹的添加程序进行判定  不存在时进行创建
    - 4.tiles里对于tcg8.svs位置的获取需要重新更改 
    - 5.tiles里有对于svs图源的冗余操作
    - 6.删除冗余程序段解决了tiles tcg8 获取报错问题
    - 7.图标在不同分辨率下的自适应
    - 8.分多一个进程去做背景去除
    - 9.在进程执行过程中添加进度条的判断
    - 10.背滤程序调试阶段 通过二值化的矩阵进行处理时需要在边界处进行多填一个数的操作
         减少在进行取整操作时的边界漏数错误
    - 11.注意系统行列顺序关系  本系统为先row 再col
    - 12.最开始的闪退问题是因为数据格式有问题  为字符串
    - 13.字符集设为utf-8
    - 14.分割结果预览窗口因需要实例化无法进行多进程创建


###2021.5.06
-  主要任务： 
    - 1.在背景中提取亮目标，TRIANGLE法优于OTSU法，而在亮背景中提取暗目标，OTSU法优于TRIANGLE法。
    - 2.lmdb的打开支持多进程同时进行读取操作
    - 3.pool.imap方式行不通尝试使用easygui
    - 4.采用动态阈值后的锐化处理得到了更好的效果
    - 5.多进程任务进程空间开辟后怎么关闭节省内存占用
    - 6.成功完成数据保存  进行性能比较  更改lmdb数据的存储形式
    - 7.评估qpixmap处理数据的方式尝试数字化转存
    - 8.TileGraphicsitem为图元初始化操作类
    - 9.合并两个path  集中lmdb的单张切片保存
    - 10.在弹出窗中dialog和widget可以为父子嵌套关系
    - 11.针对轮廓的查找方式可以进行性能对比  
              # contours, hirarchy = cv2.findContours(iOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
              contours, hirarchy = cv2.findContours(iClose, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    - 12.nparray的dtype警告
    - 13.PIL与opencv颜色通道转换
    - 14. 软件mask绘图框架为QPainter只能绘制几何图形   寻找绘制边缘图形mask的可能
    - 15. 轮廓结果为一个多元素数组每一个封闭轮廓为一个数组
    - 16. drawpainterpath  drawpoints()
    - 17. 除了external 别的都是所有轮廓 用list 可以不分层级比较推荐
    - 18. 进行tiles计算 算的不是概率是找到tum类型的图片地址
    - 19. format()里处理字符串
    - 20. mainwidow类子窗口传参加self
    - 21. 将历次框选结果细胞分割保存
    
    

5.14 1.完成了预览切片的文件名对应显示  封装好了右键功能  图片的鼠标捕捉还没写好
     2.之前采用字符串的形式保存图片到lmdb 还原时程序时间复杂度不够好 想要优化性能
       尝试了数字形式读取时转换成qimage的形式搭建完后存在溢出和报错 正在解决
     3.因为切片数量太多 现在更改为针对svs图片创建对应的lmdb 方便检索文件名查找 存放在之前的data-index文件夹中
     4.更改程序时出现了预览弹窗闪退问题 现在已解决
   
5.15 1.通过 自建预览窗口的 图片显示label 的方式完成了在预览窗口里的tiles捕捉
       保存感兴趣tiles  并进行了特异地址指定
     2.调研lmdb写入 直接保存数组很容易溢出 而且运行时间变慢 还是采用原来的.encode保存方式
     3.使用了qupath分割细胞功能 尝试看看代码逻辑进行借鉴 
     4.使用了imagej


5.17 1.为匹配细胞分割 对之前的区域标记和区域框选功能进行优化 
     2.查找UNET相关内容
     3.尝试完成动态边缘分割功能 单一像素点的绘制宽度视觉效果太差  
     4.思考细胞核的图像分割计数

5.18 1. 做完了传统opencv方法的细胞分割demo 正在往平台里整合框选保存显示功能
     2. 细胞分割的细胞核边缘分割计数存在瑕疵正在优化
     3. 下了一个别人用的unet数据集 后面准备试一下
     
5.19 1. 在获取轮廓的时候出现了nparray的dtype警告已解决
     2. 针对不同自适应方式adaptiveThreshold和轮廓查找方式findContours进行了测试 现在框选轮廓效果尚可
     3. 将框选按键弹窗画轮廓实现
     4. 在进行动态mask绘制的尝试  因坐标系和qpainter的限制存在一些问题 正在做
     
5.20 1.针对不同轮廓近似方式进行评估优化 
     2.测试drawimage drawpoints drawpixmap的不同方式 思考实现方式
     3.正在进行多点绘制MASK的功能实现  在SVS层级变换下保证实时刷新坐标不太好对应
     4.绘点不行的话 就对获取到的封闭轮廓采取循环遍历的方式进行尝试

5.21 1.原图上通过尝试drawimage drawpoints drawpixmap的不同方式
       进行传参时对轮廓运算结果的数据结构有些误解用了比较多时间重组数组构成QPoints类型数据
       最终通过points坐标对应方式实现了绘制mask 但只实现了原图最后一层
     2.当前情况时间复杂度较高需要进一步优化  在各个层级间无缝的刷新呈现还有瑕疵正在改

5.22 1. 通过实施捕捉框选区域以及刷新绘制的方法 实现了层级缩放时的轮廓像素点级的MASK绘制
     2. 重绘层级时的闪退问题是因为MASK变量绘制时的Zvalue冲突 已经解决 可以在任意层级显示结果 
     3. 正在写动态阈值获取细胞核的方式之前固定值的方式存在瑕疵无法准确划分紧凑的细胞团 完成计数 在写数据可视化的二维绘图demo
     4. 构思与九分类结果的切片整合的方式 完成这个框架后进行UNET的置入

5.23 1. 完善了一下重复框选时的操作逻辑和绘制Mask时的QPen设置
     2. 为完成计数对二值化后的阈值获取进行不同的尝试  领域内均值 像素点加权和高斯  分别在不同图上进行评估
        各有利弊 自适应方式容易将深色非核区域错认为细胞核 这部分还在尝试找方法
     3. 现在写了计数函数加入了闭合区域面积限制  做进一步筛除优化效果 并对原图MASK的绘制进行了优化
        闭合区域面积的限制阈值还存在优化空间
     4. 后续针对计数结果统计绘图

5.24 1. 优化了opencv找轮廓的方法 采用先高斯尽可能多的选区域然后算区域内闭合面积均值算阈值的自适应方法对获得轮廓进行筛选和计数
     2. 做深度网络unet的细胞分割的部分功能 查阅资料和代码  正在进行实现

5.25 1. 和小芳讨论unet实现  讨论mask的原图插值
     2. 用于做细胞分割还是需要自己重新训练之前找到的的数据集格式有问题
        现在正在找其他的 找到一个用tensorflow实现的准备转成pytorch 正在改 这一个的数据集下下来存在一点小问题
        明天争取解决进行网络训练
     3. 解决了建立表格qchart无法识别导包的问题

5.26 1. 继续找关于unet的细胞检测实现 kaggle 2018 data science bowl 竞赛发现了合适的数据 正在写这个实现方式是只能进行256*256的切片操作
        准备训练完后再看
     2. 另一个效果较好需要改写的github项目  看了作者文章没有提供数据集  发了封邮件问数据集相关的还没回
        他的方法可以实现任意尺寸图片 测试用例里并没有实现方式  正在研究这个地方

5.27 1. 调试代码针对下好的竞赛数据进行网络训练 出现了cudart64_110.dll警告
     2. 昨天的邮件还未收到回复
     3. 增加细胞分割时的传参 解决了在图元viewer层级弹窗传参时的闪退问题 
     4. 当前的训练结果不太好且无法计算任意尺寸 决定将之前找到的别人文章里效果较好的网络进行置入并进行细胞统计时的阈值优化
     
5.29 1.完成了unet的置入可以实时刷新细胞检测mask 弹窗选择分割方式
       质量不好的图细小细胞核还是效果不好 比之前效果要好
     2.正在写计数优化部分后面对比显示两种方式的结果 用图表进行gui

5.30 1.写了unet计数 在解决局部区域内的细胞核粘连无法正确区分的问题
     2. 正在针对九分类的结果进行 癌类tum tiles的 细胞核数检测计算 
         由于tiles较多计算效率不高 正在做加速优化
         
6.1 1. cv2从index中读tiles地址时一直报错搞了一下午试了用pil等各种方法 
       最终发现是index中换行符没去掉导致的路径无法读取问题 现在已经解决
    2. 完成了九分类中患癌子类的tiles的细胞计数 明天做图

6.2 1. 写好阈值传参 均值的超参数
    2.  做了unet里 过小 区域的去除
    3. 在写opencv里细胞核重叠的mask区分绘制 小框选区域内数字化图像的获取遇到了困难正在解决

6.3 1.通过整合unet和opencv两种方式的输出contours和centermask'
      进行统一格式的计数函数编写 并向内传参的方式 进行小核筛除和重叠区分 
    2. 在做对重叠部分区分时的轮廓绘制和计算 以及之前老师说过的色域的实现方式

6.4 1. 通过对opencv高斯方式进一步加权调整阈值的方法进一步优化轮廓划分 区分重叠部分
       因之前的方法中存在一个超大面积的外轮廓 所以取均值效果一直不好 现在已经改善
    2. 通过对神经网络的轮廓输出进行优化 将输出轮廓对应成灰度图 判断面积后递归轮廓内色深重新计算轮廓
       实现了对部分重叠的有效区分 对于狭长的单核划分效果不太好   后续继续优化
    3. 后面整合框选计数和tiles分类计数 先把结果列表gui做出来

6.5 1. 重写九分类tiles里的计数部分 将各类数量统计出来方便后面建表

6.6 1. 对已分类的多个tiles做unet计算时的多进程加速  同时传参中对细胞核位置和面积信息进行传参返回
       后面生存分析做完后在针对打包封装 对添加功能进行体积优化
    2. 学习QChart中的饼图柱形图的多页面切换和动态绘制 进行实现

6.7 1. 解决了统计切片细胞核多线程时的闪退问题 之前是torch处理时num_workers数指定的问题
    2. 整合opencv方式与unet方式进一步优化结果 写gui部分设计结果tabs

6.10 1. 因要进行ui设计把弹窗表封为window对象出错了   学习了多窗口实例化ui使用的相关内容 解决了问题
        整合数据   构思绘制内容 分为分类和框选两种呈现方式
     2. 整合opencv unet两种方式 将unet的输出用opencv将输出的灰度mask图用高斯循迹方法计算一下再封装出来
        效果一般提升不大  可以把小的误选区域去除掉 优化了计数 计数结果还可以

6.11 1. 整理软件平台输入输出 将最近新植入的功能进行打包 列提纲
     2. 打包后遇到了defaultParams 报错  为matplotlib包里的报错
     3. 解决了打包tensor时的报错问题

6.15 1. 在整合结果参数时发现计数结果不对  优化了程序
     2. 整合传参做出各类型表接口 统一ui
     3. 学习回归分析的相关内容

6.16 1. 完成了图标的gui和内置  完成了柱图堆叠图的初始化   留出设计百分比图饼图接口当前暂时用不到
     2. 几种数据的同表内整合还需要优化
     3. 在写将历次框选数据进行记忆化保存 供给绘表
 
6.17 1. 写好了弹窗显示框选核数  不够精致后续进行优化
     2. 因刷新显示导致记忆化存储框选区域传参 不太好做正在解决
     
6.18 1. 以treewidget的形式进行记忆化统计的设计 初始化传参行列时出现了卡退bug 给定目录树序号重写初始化部分后解决了bug'
     2. 在写 之前 老师说的那种以均值为核心的上下扩充最值的图表
     3. 类内记忆化存储实现了  类间还没有 有bug在改

6.19 1. 以map的形式构建key value 对应解决了 类间记忆化传参
     2. 可以记忆化存储在gui显示各次的分割结果 但窗口闪退搞了好久 mainwindow类传参还是有限制 正在解决
     3. 找了个蜡烛图的实例学习 写了demo 后面整合到平台里

6.20 1. 做了很多尝试在 主窗口下实例化子窗口maindow类 并进行ui设计后 无法实现实时记忆化刷新变量传参
        最终原因为 受到了弹出子窗口mianwindow底层show函数的限制 传参闪退
     2. 重做子窗口ui 找到了用dialog.exec的方式解决了传参问题 完成了ui自适应的部署
        后面把图和框选记忆化结果做好
        
        
6.22 1. 构建好了框选记忆化存储的目录树形式的结果显示
     2. 虚构数据做cox会回归和生存分析

6.23 1. 学习PYQT箱形图绘制 在平台里搞定了 还有细节需要优化
     2. 将回归分析的结果置入qchart还在找方法

6.24 1. 还是不太懂cox跟师兄和小芳交流生存分析的实现方式 单一病人信息没法做
     2. 后面虚构一批数据尝试一下置入平台
     
6.25 1. 找到了tcga的生存数据  正在用lifelines包和sklearn写程序 先用matplotlib做个demo出来
        遇到了表头报错和 cimp报错问题正在解决
     2. 做好了结果后散点做成qchart

6.26 1. 报错问题通过修改表头colunms和 trainfeature 得到了解决 
     2. 根据tcga的生存数据 设计导入coxphfitter做出了cox曲线 
     3. 再找数据集生存表格中各个参数的计算方法  后面对新录入的图像进行数据添加扩充数据

6.28 1. 分别作出训练和测试的km回归结果
     2. 把结果图表集成到平台里
     3. cox的结果呈现不够美观正在优化

6.29 1. 做完了cox在测试和训练数据中的横向箱图
     2. 后面把几个结果整到一起按键功能封装到平台里
 
6.30 1. 对预测结果做了整合km曲线导入模型后plot完成
     2. cox的结果从coxphfitter中整合时plot缺失一个结果 正在解决
     
7.1 1. 解决了之前的问题 把四个曲线结果都整合进去了
    2. 把数据运算方式找到整合进去

7.2 1. 尝试将数据存储list进行优化  
    2. 封装生存分析功能 正在进行整合
    
7.3 1. 封装好了计算出的结果
    2. 可以按键选择xlsx文件进行处理和结果显示做好了这个接口

7.5 封装列表读取把数据存进去

7.6 优化数据处理过程中的多进程 尝试在计算轮廓时提高效率

7.7 1.计算轮廓的多进程方式 效率提升不明显
    2.解决今天发现的小bug

7.8 1. 解决闪退bug 显示分类切片的细胞核数弹窗显示
    2. 在python环境里打包qchart时出错了 qchart不是pyqt自带包
       正在解决

7.14 1. 修改图标  优化封装功能按键的checkable 避免逻辑混乱
     2. 找python下做多进程监视且互不干扰的方式

7.15 1. 做按键弹窗表格的优化 
     2. 加入更新列表的按键选择  加速初始化避免重复计算
     3. 后面尝试将初始化软件后的载入图像记录  记忆存储后载入子窗口表格

7.16 1. 初始化一个dict 用以存储软件使用过程中的图片记录 把结果转存入表格子窗口传参
     2. 在做是否进行过结果计算的取舍

7.17 1.打开子窗口后无操作会闪退 后发现是tabs里的鼠标捕捉问题 已解决 
     2.解决了记忆化传参导致的子窗口闪退问题
     
7.19 1. 做生存分析部分的训练和测试的分割

7.20 1. 梳理平台代码 复盘功能逻辑 准备面试

7.21 1. 学习准备高并发的相关知识 后期做并发编程

7.22 1. 学习高并发 以及父子类传参  准备面试
     2. 梳理平台进行分割时的进程池创建逻辑 进行优化伪多线程机制
     
7.27 1. 优化状态栏模块的功能 

7.29 1. 重写viewparam类的参数定义

7.30 1. 整理数据库的知识点   学习redis