from PyQt5.QtWidgets import QGraphicsScene


class SlideGraphicsScene(QGraphicsScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_downsample = 1
        self.cur_xywhds = (0, 0, 0, 0, 1)
        self.dirty = True

    def reset(self):
        # reset variables
        self.cur_downsample = 1
        self.cur_xywhds = (0, 0, 0, 0, 1)
        # remove the old pixelmap
        for item in self.items():
            self.removeItem(item)
        self.dirty = True

    def paint_view(self, view, x, y, w, h, downsample=None, wmax=100000, hmax=10000):
        if downsample is None:
            downsample = self.cur_downsample

        if ((x, y, w, h, downsample) == self.cur_xywhds) and not self.dirty:
            return
        else:
            self.cur_xywhds = (x, y, w, h, downsample)

        if self.cur_downsample == downsample:
            # only update objects if downsample does not change
            # as when upon change of zoom the window changes as well
            for item in self.items():
                if hasattr(item, 'update_content'):
                    item.update_content(x, y, w, h, downsample)

        if self.cur_downsample != downsample:
            view.scale(self.cur_downsample / downsample, self.cur_downsample / downsample)
            self.cur_downsample = downsample


        self.dirty = False

##'graphicsscene的自我反应运行且没有返回值'
