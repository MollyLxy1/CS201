```python
#dijkstra模板
import heapq

def dijkstra(n, graph, start):
    """
    n: 节点数量
    graph: 邻接表，格式为 {u: [(v, weight)]}
    start: 起始节点
    返回从 start 出发到各个节点的最短路径长度数组 dist
    """
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]  # (距离, 节点)

    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dist

#变形
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def min_cost_to_travel(m, n, grid, start, end):
    dist = [[float('inf')] * n for _ in range(m)]
    if grid[start[0]][start[1]] == '#' or grid[end[0]][end[1]] == '#':
        return "NO"
    # 使用优先队列存储当前的位置和体力消耗
    pq = []
    heapq.heappush(pq, (0, start[0], start[1]))  # (体力消耗, x, y)
    dist[start[0]][start[1]] = 0

    while pq:
        cost, x, y = heapq.heappop(pq)
        if (x, y) == end:
            return cost

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] != '#':
                new_cost = cost + abs(int(grid[nx][ny]) - int(grid[x][y])) if grid[nx][ny] != '#' else float('inf')

                # 如果找到更少消耗的路径，更新
                if new_cost < dist[nx][ny]:
                    dist[nx][ny] = new_cost
                    heapq.heappush(pq, (new_cost, nx, ny))

    return "NO"
m, n, p = map(int, input().split())
grid = []
for _ in range(m):
    grid.append(input().split())
for _ in range(p):
    sx, sy, ex, ey = map(int, input().split())
    result = min_cost_to_travel(m, n, grid, (sx, sy), (ex, ey))
    print(result)
#变形    
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # 建图：邻接表
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))

        # 最短路径字典，记录每个节点被首次到达的最短时间
        dist = dict()

        # 小根堆，存储的是 (到达时间, 节点)
        heap = [(0, k)]

        while heap:
            time, node = heapq.heappop(heap)
            if node in dist:
                continue  # 已访问，跳过

            dist[node] = time
            for nei, wt in graph[node]:
                if nei not in dist:
                    heapq.heappush(heap, (time + wt, nei))

        # 如果并非所有节点都被访问，说明有节点无法到达
        if len(dist) != n:
            return -1
        return max(dist.values())
```

```python
def bellman_ford(n, edges, start):
    """
    n: 节点数量
    edges: 边集列表 [(u, v, w)]
    start: 起始节点
    返回从start出发到各个节点的最短路径长度数组dist
    如果图中存在负权环，则返回None
    """
    dist = [float('inf')] * n
    dist[start] = 0
    
    for i in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                
    # 检查负权环
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # 存在负权环
            
    return dist
```

```python
def floyd_warshall(n, graph):
    """
    n: 节点数量
    graph: 邻接矩阵，graph[u][v] 表示从节点u到节点v的权重
    返回所有节点对之间的最短路径长度矩阵
    如果图中存在负权环，则返回None
    """
    dist = [[float('inf')]*n for _ in range(n)]
    
    # 初始化
    for i in range(n):
        dist[i][i] = 0
    for u in range(n):
        for v, w in graph[u]:
            dist[u][v] = w
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    
    # 检查负权环
    for i in range(n):
        if dist[i][i] < 0:
            return None  # 存在负权环
            
    return dist
```

- **Dijkstra算法**适用于边权为非负数的图。
- **Bellman-Ford算法**能检测是否存在负权环，如果存在负权环且环可到达目标节点，则无法定义最短路径。
- **Floyd-Warshall算法**是一个经典的动态规划算法，用于计算所有节点对之间的最短路径，适用于稠密图或需要查询多对节点间的最短路径情况。

```python
from collections import deque,defaultdict
def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
    n = len(colors)
    graph = defaultdict(list)
    indegree = defaultdict(list)
    indegree = [0]*n

    for u,v in edges:
        graph[u].append(v)
        indegree[v] += 1

    dp = [[0] * 26 for _ in range(n)]
    result = 0
    visited = 0

    queue = deque([i for i in range(n) if indegree[i] == 0])
    while queue:
        node = queue.popleft()
        visited += 1
        # 更新当前节点的颜色计数
        color_index = ord(colors[node]) - ord('a')
        dp[node][color_index] += 1
        result = max(result, dp[node][color_index])

        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            for i in range(26):
                dp[neighbor][i] = max(dp[neighbor][i], dp[node][i])

            if indegree[neighbor] == 0:
                queue.append(neighbor)
    return result if visited == n else -1
```

拓扑排序+dp

```python
from collections import defaultdict

def has_cycle_dfs(n, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * n
    def dfs(node, parent):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    for i in range(n):
        if not visited[i]:
            if dfs(i, -1):
                return True
    return False
```

无向图判断是否有环

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False  # 已连接，不能再连接（避免成环）
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True  # 成功合并

def kruskal(n, edges):
    """
    Kruskal 算法求最小生成树
    参数:
        n: 顶点个数
        edges: 边列表，每条边为 (权重, u, v)
    返回:
        最小生成树的权重之和
    """
    # 按权重排序
    edges.sort()
    uf = UnionFind(n)
    mst_weight = 0
    edges_used = 0

    for weight, u, v in edges:
        if uf.union(u, v):
            mst_weight += weight
            edges_used += 1
            if edges_used == n - 1:
                break  # 提前结束

    # 若边数不足 n-1，说明图不连通
    return mst_weight if edges_used == n - 1 else None
```

kruskal最小生成树（MST）适合稀疏树

```python
import heapq
def prim(n, adj):
    """
    Prim 算法求最小生成树（使用堆优化，邻接表）
    参数:
        n: 节点数
        adj: 邻接表，形如 adj[u] = [(权重, v), ...]
    返回:
        最小生成树的权重之和
    """
    visited = [False] * n     # 标记是否已加入 MST
    min_heap = [(0, 0)]       # (边权重, 节点编号)，从0号节点开始
    total_weight = 0
    count = 0                 # 加入 MST 的节点数

    while min_heap:
        weight, u = heapq.heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        total_weight += weight
        count += 1
        if count == n:
            break

        for w, v in adj[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (w, v))

    return total_weight if count == n else None  # 若图不连通，返回 None
```

prim最小生成树 适合稠密树

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # 存储子节点，键是字符
        self.is_end = False  # 标志是否是一个完整单词的结尾

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        # 插入一个单词
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True  # 标记单词结束

    def search(self, word):
        # 查找完整单词是否存在
        node = self._find_node(word)
        return node is not None and node.is_end

    def startsWith(self, prefix):
        # 判断是否有该前缀
        return self._find_node(prefix) is not None

    def _find_node(self, prefix):
        # 辅助函数：找到表示 prefix 的节点
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node
```

前缀树Trie

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorder(root):
    """前序遍历：根 -> 左 -> 右"""
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root):
    """中序遍历：左 -> 根 -> 右"""
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root):
    """后序遍历：左 -> 右 -> 根"""
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]

```

二叉树的三种顺序遍历

```python
from collections import deque

def level_order_flat(root):
    """层序遍历，返回扁平列表（如 [A, B, C, D, E, F]）"""
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return result
```

层序遍历

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def buildTree_pre_in(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_val = preorder[0]
    root = TreeNode(root_val)
    idx = inorder.index(root_val)
    root.left = buildTree_pre_in(preorder[1:1+idx], inorder[:idx])
    root.right = buildTree_pre_in(preorder[1+idx:], inorder[idx+1:])
    return root

```

前序+中序建树

```python
def buildTree_in_post(inorder, postorder):
    if not inorder or not postorder:
        return None
    root_val = postorder[-1]
    root = TreeNode(root_val)
    idx = inorder.index(root_val)
    root.left = buildTree_in_post(inorder[:idx], postorder[:idx])
    root.right = buildTree_in_post(inorder[idx+1:], postorder[idx:-1])
    return root
```

中序+后序建树

```python
def build_diff_array(nums):
    """
    构建差分数组
    输入:
        nums: 原始数组
    输出:
        diff: 差分数组
    """
    n = len(nums)
    diff = [0] * n
    diff[0] = nums[0]  # 差分数组第一个元素等于原数组第一个元素
    for i in range(1, n): # 差分数组的每个元素等于原数组当前位置与前一个位置的差值
        diff[i] = nums[i] - nums[i - 1]
    return diff

def increment(diff, l, r, val):
    """
    输入:
        diff: 差分数组
        l: 区间左端点（含）
        r: 区间右端点（含）
        val: 增加的值
    """
    diff[l] += val  # 左端点加 val，表示从这里开始加
    if r + 1 < len(diff):
        diff[r + 1] -= val  # 右端点后一个位置减 val，表示从这里停止加

def recover_array(diff):
    """
    输入:
        diff: 差分数组
    输出:
        res: 还原后的数组
    """
    n = len(diff)
    res = [0] * n
    res[0] = diff[0]  # 原数组第一个元素等于差分数组第一个元素
    for i in range(1, n):
        # 前缀和恢复原数组每个元素的值
        res[i] = res[i - 1] + diff[i]
    return res
```

差分数组

```python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

def insert(root, val):
    """插入节点到BST"""
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    else:  # val >= root.val，通常右子树允许等于
        root.right = insert(root.right, val)
    return root

def search(root, val):
    """在BST中查找值"""
    if not root or root.val == val:
        return root
    if val < root.val:
        return search(root.left, val)
    else:
        return search(root.right, val)

def find_min(root):
    """找BST的最小节点（左子树最左节点）"""
    while root.left:
        root = root.left
    return root

def delete_node(root, val):
    """删除BST中值为val的节点"""
    if not root:
        return None
    if val < root.val:
        root.left = delete_node(root.left, val)
    elif val > root.val:
        root.right = delete_node(root.right, val)
    else:
        # 找到节点，进行删除
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        # 两个子节点都存在，找右子树最小节点替代
        min_larger_node = find_min(root.right)
        root.val = min_larger_node.val
        root.right = delete_node(root.right, min_larger_node.val)
    return root

def inorder(root):
    """中序遍历BST，返回升序列表"""
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)
```

BST 二叉搜索树

```py
'''
为了找到最大的非空子矩阵，可以使用动态规划中的Kadane算法进行扩展来处理二维矩阵。
基本思路是将二维问题转化为一维问题：可以计算出从第i行到第j行的列的累计和，
这样就得到了一个一维数组。然后对这个一维数组应用Kadane算法，找到最大的子数组和。
通过遍历所有可能的行组合，我们可以找到最大的子矩阵。
'''
def max_submatrix(matrix, n):
    def kadane(arr):
      	# max_ending_here 用于追踪到当前元素为止包含当前元素的最大子数组和。
        # max_so_far 用于存储迄今为止遇到的最大子数组和。
        max_end_here = max_so_far = arr[0]
        for x in arr[1:]:
          	# 对于每个新元素，我们决定是开始一个新的子数组（仅包含当前元素 x），
            # 还是将当前元素添加到现有的子数组中。这一步是 Kadane 算法的核心。
            max_end_here = max(x, max_end_here + x)
            max_so_far = max(max_so_far, max_end_here)
        return max_so_far

    max_sum = float('-inf')

    for top in range(n):
        temp_col_num = [0] * n
        for bottom in range(top, n):
            for col in range(n):
                temp_col_num[col] += matrix[bottom][col]
            max_sum = max(max_sum, kadane(temp_col_num))
    return max_sum

# 输入处理
import sys
data = sys.stdin.read().split()
n = int(data[0])
numbers = list(map(int, data[1:]))
matrix = [numbers[i * n:(i + 1) * n] for i in range(n)]

max_sum = max_submatrix(matrix, n)
print(max_sum)
```

kadane（二位变种）

卡特兰数

1. **二叉搜索树（BST）数目**
   - 对于给定的 `n` 个不同节点，能构造多少种不同形态的二叉搜索树。
   - 数量等于第 `n` 个卡特兰数。
2. **二叉树的形态数**
   - 所有含有 `n` 个节点的不同形态的二叉树数量。
3. **合法括号序列的个数**
   - 长度为 `2n` 的括号序列中，合法配对的序列个数。
4. **山脉序列数量**
   - 长度为 `2n` 的序列，其上升和下降方式满足特定约束的序列数量。
5. **凸多边形三角剖分数**
   - 一个凸多边形有多少种不同的三角剖分方法。
6. **栈操作序列数**
   - 长度为 `2n` 的进栈和出栈操作序列数，保证栈不空且合理。
7. **完全二叉树的计数**

```python
from math import comb
def catalan_formula(n):
    """
    计算第 n 个卡特兰数，使用组合数公式
    C_n = (1/(n+1)) * C(2n, n)
    """
    return comb(2 * n, n) // (n + 1)

```

```python
def evalRPN(tokens):
    stack = []
    for token in tokens:
        if token in "+-*/":
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                # 注意：题目要求向 0 截断
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    return stack[0]
```

逆序波兰表达式求值

```python
from typing import List
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        heights = [0] + heights + [0]
        res = 0
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                res = max(res, h * w)
            stack.append(i)
        return res

if __name__ == '__main__':
    s = Solution()
    print(s.largestRectangleArea([2,1,5,6,2,3]))
```

柱状图中最大的矩形（stack）

```python
#输入分为两行
#第一行为一个整数N，代表二叉树中节点的个数。
#第二行为一个N个非负整数。第i个数代表二叉树中编号为i的节点上的宝藏价值。
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def main():
    N = int(input())
    values = list(map(int, input().split()))
    if N == 0:
        print(0)
        return

    # 构建完全二叉树
    nodes = [TreeNode(val) for val in values]
    for i in range(N):
        left_idx = 2 * (i + 1) - 1
        right_idx = 2 * (i + 1)
        if left_idx < N:
            nodes[i].left = nodes[left_idx]
        if right_idx < N:
            nodes[i].right = nodes[right_idx]

    def dfs(node):
        if not node:
            return (0, 0)
        left = dfs(node.left)
        right = dfs(node.right)
        # 不选当前节点
        not_rob = max(left[0], left[1]) + max(right[0], right[1])
        # 选当前节点
        rob = node.val + left[0] + right[0]
        return (not_rob, rob)

    result = dfs(nodes[0])
    print(max(result[0], result[1]))
```

树状dp（宝藏二叉树）

```python
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True  
    if n % 2 == 0 or n % 3 == 0:
        return False  
    # 6k±1 优化：所有质数 > 3 的形式必是 6k±1
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

itertools

### 1. `product`：笛卡尔积（排列）

```python
from itertools import product
print(list(product('AB', repeat=2)))
# 输出: [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]
```

### 2. `permutations`：排列

```py
from itertools import permutations
print(list(permutations([1, 2, 3], 2)))
# 输出: [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
```

### 3. `combinations`：组合（无重复）

```py
from itertools import combinations
print(list(combinations([1, 2, 3], 2)))
# 输出: [(1, 2), (1, 3), (2, 3)]
```

### 4. `accumulate`：累加

```py
from itertools import accumulate
print(list(accumulate([1, 2, 3, 4])))
# 输出: [1, 3, 6, 10]
```

### 5. `chain`：拼接多个可迭代对象

```py
from itertools import chain
print(list(chain([1, 2], [3, 4])))
# 输出: [1, 2, 3, 4]
```