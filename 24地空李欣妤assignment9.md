# Assignment #9: Huffman, BST & Heap

Updated 1834 GMT+8 Apr 15, 2025

2025 spring, Complied by <mark>李欣妤、地空学院</mark>



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### LC222.完全二叉树的节点个数

dfs, https://leetcode.cn/problems/count-complete-tree-nodes/

思路：直接递归很简单

优化的话利用完全二叉树的性质，左右子树至少有一个是满二叉树，可以直接得出节点数目。学习了一下二进制运算符（满二叉树的节点数为 `2^h - 1`，其中 `h` 是树的高度。使用左移运算符可以高效地计算 `2^h`）

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        leftnum = self.countNodes(root.left)
        rightnum = self.countNodes(root.right)
        return 1+leftnum +rightnum
#以下是利用完全二叉树性质的解法
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        left_height = self.get_height(root.left)
        right_height = self.get_height(root.right)
        
        if left_height == right_height:
            # 左子树是满二叉树
            return (1 << left_height) + self.countNodes(root.right)
        else:
            # 右子树是满二叉树
            return (1 << right_height) + self.countNodes(root.left)
    
    def get_height(self, node):
        height = 0
        while node:
            height += 1
            node = node.left
        return height
```

> 核心逻辑
>
> 在完全二叉树中：
>
> 如果 left_height == right_height，则说明左子树是满二叉树。
> 如果 left_height != right_height，则说明右子树是满二叉树。
> 这是因为：
>
> 完全二叉树的节点从左到右依次填满，所以如果左右子树的高度相等，左子树必然是满二叉树。
> 如果左右子树的高度不相等，则右子树必然是满二叉树（因为右子树的高度比左子树少一层）。



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250417202015668](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250417202015668.png)



### LC103.二叉树的锯齿形层序遍历（20min）

bfs, https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

思路：使用deque存储节点（也是bfs经典处理了），定义一个bool来判断遍历方向，在用一个deque实现



代码：

```python
from collections import deque

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        result = []
        queue = deque([root])
        l_to_r = True  # 第一层从左到右
        while queue:
            level_size = len(queue)
            current_level = deque()
            for _ in range(level_size):
                cur = queue.popleft()
                if l_to_r:
                    current_level.append(cur.val)
                else:
                    current_level.appendleft(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            result.append(list(current_level))


            l_to_r = not l_to_r
        return result                        
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250417214441051](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250417214441051.png)



### M04080:Huffman编码树（20min）

greedy, http://cs101.openjudge.cn/practice/04080/

思路：看了课件才明白题目在说什么



代码：

```python
import heapq
def find_min_weightsum(weights):
    result = 0
    heapq.heapify(weights)
    while len(weights)>1:
        a = heapq.heappop(weights)
        b = heapq.heappop(weights)
        result += a+b
        heapq.heappush(weights,a+b)
    return result
n = int(input())
weights = list(map(int,input().split()))
print(find_min_weightsum(weights))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250417221038286](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250417221038286.png)



### M05455: 二叉搜索树的层次遍历（30min）

http://cs101.openjudge.cn/practice/05455/

思路：还是感觉最重要的是读懂题目，层序遍历是已经会的模板了，在学习一个插入构建二叉搜索树的模板

debug中学到了用set去重不能保证原始顺序，可以用dict.fromkeys( )

构建二叉搜索树（BST）的步骤如下：

### 1. **理解二叉搜索树的定义**
   - 每个节点最多有两个子节点（左子节点和右子节点）。
   - 对于任意节点：
     - **左子树**的所有节点值 **小于** 该节点的值。
     - **右子树**的所有节点值 **大于** 该节点的值。
   - 没有重复的节点值（除非特别允许）。

### 2. **构建步骤**
   - **初始状态**：树为空（根节点为 `null`）。
   - **插入规则**：
     1. 从根节点开始比较。
     2. 如果待插入值 **小于** 当前节点值，递归插入到左子树。
     3. 如果待插入值 **大于** 当前节点值，递归插入到右子树。
     4. 如果当前节点为 `null`，则在此位置创建新节点。

### 3. **示例代码（Python）**
```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def insert(root, val):
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    elif val > root.val:
        root.right = insert(root.right, val)
    return root  # 如果值已存在，直接返回（假设无重复）

def build_bst(data):
    root = None
    for num in data:
        root = insert(root, num)
    return root

# 示例数据
data = [10, 5, 15, 2, 7, 12, 18]
bst_root = build_bst(data)
```

### 4. **构建过程演示（以示例数据为例）**
   - 插入 `10`：根节点。
   - 插入 `5`：成为 `10` 的左子节点。
   - 插入 `15`：成为 `10` 的右子节点。
   - 插入 `2`：与 `10` 比较 → 左到 `5`，再成为 `5` 的左子节点。
   - 插入 `7`：与 `10` 比较 → 左到 `5`，再成为 `5` 的右子节点。
   - 插入 `12`：与 `10` 比较 → 右到 `15`，再成为 `15` 的左子节点。
   - 插入 `18`：与 `10` 比较 → 右到 `15`，再成为 `15` 的右子节点。

最终树结构：
```
      10
     /  \
    5    15
   / \  / \
  2  7 12 18
```

### 5. **注意事项**
   - **输入顺序影响树形**：数据插入顺序不同可能导致树的高度不同（如升序插入会退化成链表）。
   - **时间复杂度**：
     - 平均情况：插入每个节点是 \(O(\log n)\)，总复杂度 \(O(n \log n)\)。
     - 最坏情况（有序数据）：\(O(n^2)\)。
   - **优化**：如果数据预先排序，可采用二分法递归构建平衡BST（如AVL树或红黑树）。

### 6. **验证BST是否合法**
可通过中序遍历检查是否得到升序序列：
```python
def inorder_traversal(root):
    return inorder_traversal(root.left) + [root.val] + inorder_traversal(root.right) if root else []
print(inorder_traversal(bst_root))  # 应输出升序列表
```

通过以上步骤，即可从任意数据集构建二叉搜索树。



代码：

```python
from collections import deque

class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def insert(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    elif val > root.val:
        root.right = insert(root.right, val)
    return root

def level_order(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        result.append(str(node.val))
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

nums = list(map(int, input().split()))
unique_nums = list(dict.fromkeys(nums))
root = None
for num in unique_nums:
    root = insert(root, num)
print(' '.join(level_order(root)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250417223249950](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250417223249950.png)



### M04078: 实现堆结构（45min）

手搓实现，http://cs101.openjudge.cn/practice/04078/

类似的题目是 晴问9.7: 向下调整构建大顶堆，https://sunnywhy.com/sfbj/9/7

思路：最小堆是一种完全二叉树，其中每个父节点的值都小于或等于其子节点的值。我们使用数组来表示堆，并通过上浮（插入时）和下沉（删除时）操作来维护堆的性质

思路不难，一步一步写出来比较复杂，还是靠ai老师帮忙改代码了啊



代码：

```python
import sys
class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, num):
        self.heap.append(num)
        self._sift_up(len(self.heap) - 1)

    def extract_min(self):
        if not self.heap:
            return None
        min_val = self.heap[0]
        last_val = self.heap.pop()
        if self.heap:
            self.heap[0] = last_val
            self._sift_down(0)
        return min_val

    def _sift_up(self, idx):
        parent = (idx - 1) // 2
        while idx > 0 and self.heap[idx] < self.heap[parent]:
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            idx = parent
            parent = (idx - 1) // 2

    def _sift_down(self, idx):
        left = 2 * idx + 1
        right = 2 * idx + 2
        smallest = idx
        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right
        if smallest != idx:
            self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
            self._sift_down(smallest)

input_lines = sys.stdin.read().split()
ptr = 0
n = int(input_lines[ptr])
ptr += 1
heap = MinHeap()
for _ in range(n):
    type_op = int(input_lines[ptr])
    ptr += 1
    if type_op == 1:
        u = int(input_lines[ptr])
        ptr += 1
        heap.insert(u)
    elif type_op == 2:
        print(heap.extract_min())
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422163007259](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250422163007259.png)



李欣妤、24地空学院

### T22161: 哈夫曼编码树

greedy, http://cs101.openjudge.cn/practice/22161/

思路：好难的手搓，自己试图写了一半就放弃了，不知道写这么长的代码要怎么坚持下去，莫名对程序员这个职业肃然起敬

## 2. 学习总结和收获

写了大约五道leetcode树的题目，不用自己建树方便很多

本周的作业两个手搓题目挺复杂，其他的题目能够把题目看懂就比较容易实现。利用了一些数据结构的性质比如说完全二叉树、平衡二叉树、二叉搜索树等等，感觉对树的题目的熟练度有在上升，接下来要开始图了，保持再多写一点题目吧









