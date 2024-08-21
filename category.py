from typing import *
from collections import OrderedDict


class Category:
    def __init__(self, name: str):
        self.name = name
        self.depth = 0
        self.parent = None
        self.childs = OrderedDict()
        self.keywords = set()
        self.documents = set()
        self.category_list = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: 'Category'):
        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            return False

    def __repr__(self):
        if len(self.childs) == 0:
            return self.name + ": []"
        return self.name + ": " + str(list(self.childs.keys()))

    def get_hierarchical_path(self) -> List[str]:
        # path not including 'ROOT'
        path = []
        parent = self.parent
        while parent.name != "ROOT":
            path.insert(0, parent.name)
            parent = parent.parent
        path.append(self.name)
        return path

    def set_depth(self, depth: int) -> None:
        self.depth = depth

    def get_depth(self) -> int:
        return self.depth

    @staticmethod
    def get_max_depth(root_node: 'Category', cur_depth: int = 0) -> int:
        if len(root_node.get_childs()) == 0:
            return cur_depth
        cur_depth_copy = cur_depth
        for child in root_node.childs:
            child_depth = Category.get_max_depth(child, cur_depth_copy + 1)
            if child_depth > cur_depth:
                cur_depth = child_depth
        return cur_depth

    def set_parent(self, parent: 'Category') -> None:
        self.parent = parent

    def add_child(self, child: 'Category', depth: int) -> None:
        child.set_parent(self)
        child.set_depth(depth)
        self.childs[child] = depth

    def set_keywords(self, keywords: Set[str]) -> None:
        self.keywords = keywords

    def get_parent(self) -> 'Category':
        return self.parent

    def get_childs(self) -> List['Category']:
        return list(self.childs.keys())

    def get_category_list(self) -> List[str]:
        return self.category_list

    def set_category_list(self, category_list: List[str]):
        self.category_list = category_list

    def add_document(self, document_index: int) -> None:
        self.documents.add(document_index)

    def get_documents(self) -> Set[int]:
        return self.documents

    def get_documents_size(self) -> int:
        return len(self.documents)

    def find_document(self, document_index: int) -> Optional['Category']:
        if document_index in self.documents:
            return self
        else:
            for child in self.childs:
                category = child.find_document(document_index)
                if category is not None:
                    return category
            return None

    def find_category(self, categorie_path_list: List[str]) -> Optional['Category']:
        if len(categorie_path_list) == 0:
            return None

        categories_copy = categorie_path_list.copy()
        if self.name == "ROOT":
            categories_copy.insert(0, "ROOT")

        if self.name == categories_copy[0]:
            if len(categories_copy) == 1:
                return self
            for child in self.childs:
                found = child.find_category(categories_copy[1:])
                if found is not None:
                    return found
        return None

    def add_category(self, categories: List[str]) -> 'Category':
        depth = len(categories)
        if depth == 0:
            return self

        leaf_category = self.find_category(categories)
        if leaf_category:
            return leaf_category

        category_path = categories[:-1]
        leaf_category = Category(categories[-1])
        parent = self.find_category(category_path)
        if parent is None:
            parent = self.add_category(category_path)
        parent.add_child(leaf_category, depth)

        return leaf_category

    def find_category_by_word(self, word: str) -> Optional['Category']:
        for child in self.childs:
            categoty = child.find_category_by_word(word)
            if categoty is not None:
                return categoty

        if word in self.keywords:
            return self

        return None
