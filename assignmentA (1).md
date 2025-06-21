# Assignment #A: Graph starts

Updated 1830 GMT+8 Apr 22, 2025

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

### M19943:图的拉普拉斯矩阵

OOP, implementation, http://cs101.openjudge.cn/practice/19943/

要求创建Graph, Vertex两个类，建图实现。

思路：oop写起来还是比较麻烦，不过强撑着写完了（



代码：

```python
class Vertex:
    def __init__(self, id):
        self.id = id          # 顶点编号
        self.degree = 0       # 顶点的度数

    def increment_degree(self):
        """增加顶点的度数"""
        self.degree += 1


class Graph:
    def __init__(self, n):
        self.n = n  # 图的顶点数
        self.vertices = [Vertex(i) for i in range(n)]  # 初始化所有顶点
        self.adj_matrix = [[0] * n for _ in range(n)]  # 初始化邻接矩阵

    def add_edge(self, a, b):
        """添加一条无向边"""
        self.adj_matrix[a][b] = 1
        self.adj_matrix[b][a] = 1
        self.vertices[a].increment_degree()
        self.vertices[b].increment_degree()

    def laplacian(self):
        """计算拉普拉斯矩阵"""
        laplacian_matrix = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                if i == j:
                    row.append(self.vertices[i].degree)
                else:
                    row.append(-self.adj_matrix[i][j])
            laplacian_matrix.append(row)
        return laplacian_matrix


# 主程序
if __name__ == "__main__":
    # 输入顶点数和边数
    n, m = map(int, input().split())
    graph = Graph(n)  # 创建图实例

    # 添加边
    for _ in range(m):
        a, b = map(int, input().split())
        graph.add_edge(a, b)

    # 计算拉普拉斯矩阵
    laplacian_matrix = graph.laplacian()

    # 输出结果
    for row in laplacian_matrix:
        print(" ".join(map(str, row)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250425164247357](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250425164247357.png)



### LC78.子集

backtracking, https://leetcode.cn/problems/subsets/

思路：本来想开一个bool表，让ai看了一下发现二进制位运算真好用



代码：

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
            n = len(nums)
            result = []
            
            # 遍历从0到2^n - 1的所有数字
            for i in range(2**n):
                subset = []
                # 检查i的每一位是否为1
                for j in range(n):
                    if i & (1 << j):  # 如果第j位是1，则将nums[j]加入子集
                        subset.append(nums[j])
                result.append(subset)
            
            return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250425170454175](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250425170454175.png)



### LC17.电话号码的字母组合

hash table, backtracking, https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

思路：



代码：

```python
from typing import List

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        # 定义电话号码与字母的映射关系
        phone_map = ("", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz")
        
        # 如果输入为空，直接返回空列表
        if not digits:
            return []
        
        # 初始化结果列表
        ans = []
        n = len(digits)
        path = [''] * n
        
        # 深度优先搜索
        def dfs(i: int) -> None:
            # 如果已经处理完所有数字，将当前路径加入结果
            if i == n:
                ans.append(''.join(path))
                return
            
            # 获取当前数字对应的字母集合
            for c in phone_map[int(digits[i])]:
                path[i] = c  # 设置当前位为某个字母
                dfs(i + 1)   # 递归处理下一位
        
        # 从第0位开始搜索
        dfs(0)
        return ans       
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>



![image-20250425200303279](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250425200303279.png)

### M04089:电话号码

trie, http://cs101.openjudge.cn/practice/04089/

思路：字符串的sort可以大大简化查找，只需要判断相邻的是否是前缀就好了，写完看了一下trie



代码：

```python
def is_consistent(numbers):
    numbers.sort()
    for i in range(len(numbers) - 1):
        if numbers[i+1].startswith(numbers[i]):
            return False
    return True

t = int(input())
for _ in range(t):
    n = int(input())
    numbers = [input().strip() for _ in range(n)]
    print("YES" if is_consistent(numbers) else "NO")
```

以下是使用trie的代码

### 字典树的基本原理

1. **节点结构**：
   - 每个节点包含一个子节点字典 `children` 和一个标志位 `is_end_of_number`，表示当前节点是否是一个电话号码的结束位置。
2. **插入操作**：
   - 遍历电话号码的每个字符，逐层构建 Trie 树。
   - 如果在插入过程中发现某个节点已经是另一个电话号码的结尾，则说明存在前缀关系。
   - 插入完成后，如果当前节点还有子节点，则说明当前电话号码是其他号码的前缀。
3. **查询一致性**：
   - 在插入过程中检测是否存在前缀关系。如果发现冲突，直接返回 `False`。

找到了一个可视化网站：https://gallery.selfboot.cn/zh/algorithms/trie

![image-20250427153501901](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250427153501901.png)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_number = False
        #TrieNode具有的两个性质：1.有孩子（构成一棵树）
        # 2。是否是一个电话号码的结尾（判断是否为另一个号码地前缀

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self,number):
        """
        向trie中插入一个电话号码
        :param number: 要插入地电话号码
        :return: 具有一致性则返回True，否则False
        """
        node = self.root
        for letter in number:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]
        #插入完成后标记当前节点为number的结尾
            if node.is_end_of_number:
                return False
        node.is_end_of_number = True

        return len(node.children) == 0
def is_consistent(phone_numbers):
    trie = Trie()
    for number in phone_numbers:
        if not trie.insert(number):
            return False
    return True
t = int(input())
result = []
for _ in range(t):
    n = int(input())
    phone_numbers = [input().strip()for _ in range(n)]
    if is_consistent(phone_numbers):
        result.append('YES')
    else:
        result.append('NO')
print('\n'.join(result))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250427153516801](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250427153516801.png)



### T28046:词梯

bfs, http://cs101.openjudge.cn/practice/28046/

思路：先建立图把相差一个字母的单词都连起来，再bfs找最短路径



代码：

```python
from collections import deque, defaultdict
def build_graph(words):
    """
    构建图：邻接表表示。
    :param words: 单词列表
    :return: 图的邻接表
    """
    word_set = set(words)  # 将单词存入集合，方便快速查找
    graph = defaultdict(list)

    for word in words:
        for i in range(len(word)):
            # 替换单词的第 i 个字符为通配符 '*'
            pattern = word[:i] + '*' + word[i + 1:]
            # 将当前单词加入到该模式对应的列表中
            graph[pattern].append(word)

    # 构建实际的邻接表
    adj_list = defaultdict(list)
    for neighbors in graph.values():
        # 同一个模式下的单词两两相连
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                adj_list[neighbors[i]].append(neighbors[j])
                adj_list[neighbors[j]].append(neighbors[i])

    return adj_list


def find_shortest_path(start, end, adj_list):
    """
    使用 BFS 找到从 start 到 end 的最短路径。
    :param start: 起点单词
    :param end: 终点单词
    :param adj_list: 图的邻接表
    :return: 最短路径（空格分隔的字符串），如果不存在返回 "NO"
    """
    if start == end:
        return start

    queue = deque([(start, [start])])  # 队列中存储 (当前单词, 路径)
    visited = set()  # 记录已访问的单词
    visited.add(start)

    while queue:
        current_word, path = queue.popleft()

        # 遍历当前单词的所有邻居
        for neighbor in adj_list[current_word]:
            if neighbor == end:
                return " ".join(path + [neighbor])  # 找到终点，返回路径

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return "NO"  # 如果没有找到路径


n = int(input()) 
words = [input().strip() for _ in range(n)]  
start, end = input().strip().split()  
adj_list = build_graph(words)
result = find_shortest_path(start, end, adj_list)
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250427195233122](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250427195233122.png)



### T51.N皇后

backtracking, https://leetcode.cn/problems/n-queens/

思路：八皇后promax，又在leetcode上学习回溯了



代码：

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        m = n * 2 - 1
        ans = []
        col = [0] * n
        on_path, diag1, diag2 = [False] * n, [False] * m, [False] * m
        def dfs(r: int) -> None:
            if r == n:
                ans.append(['.' * c + 'Q' + '.' * (n - 1 - c) for c in col])
                return
            for c, on in enumerate(on_path):
                if not on and not diag1[r + c] and not diag2[r - c]:
                    col[r] = c
                    on_path[c] = diag1[r + c] = diag2[r - c] = True
                    dfs(r + 1)
                    on_path[c] = diag1[r + c] = diag2[r - c] = False  # 恢复现场
        dfs(0)
        return ans优化代码
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250427201405194](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250427201405194.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

图还是不熟什么graph，vertex啥的

发现树有好多好多种类型，又学到新东西啦









