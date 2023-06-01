from typing import List

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsItemGroup


class LeveledGraphicsItemGroup(QGraphicsItemGroup):
    def __init__(self, levels: List[int], parent=None):
        super().__init__(parent)
        self.levels = levels
        self.level_group = {}
        for level in levels:
            group = QGraphicsItemGroup(self)
            group.setVisible(False)
            self.level_group[level] = group
        self.visible_level = None
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setAcceptHoverEvents(False)

        # self.setFlag(QGraphicsItem.ItemHasNoContents, True)
        # self.setFlag(QGraphicsItem.ItemContainsChildrenInShape, True)
        # self.setFlag(QGraphicsItem.ItemClipsChildrenToShape, True)
        # self.setFlag(QGraphicsItem.ItemClipsToShape, True)

    def boundingRect(self):
        if self.visible_level:
            return self.level_group[self.visible_level].boundingRect()
        else:
            return QRectF()

    def add_item_to_level_group(self, level: int, item: QGraphicsItem):
        self.level_group[level].addToGroup(item)

    def remove_item_from_level_group(self, level: int, item: QGraphicsItem):
        self.level_group[level].removeFromGroup(item)

    def clear_level(self, level: int):
        group = self.level_group[level]
        for item in group.childItems():
            group.removeFromGroup(item)
            group.scene().removeItem(item)

    def update_visible_level(self, visible_level: int):
        self.visible_level = visible_level
        for level in self.levels:
            group = self.level_group[level]
            group.setVisible(level == visible_level)

    def __str__(self) -> str:
        return "{}: visible_level: {}".format(
            self.__class__.__name__, self.visible_level
        )

    def __repr__(self) -> str:
        return self.__str__()
