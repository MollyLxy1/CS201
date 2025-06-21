# Assignment #6: 回溯、树、双向链表和哈希表

Updated 1526 GMT+8 Mar 22, 2025

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

### LC46.全排列

backtracking, https://leetcode.cn/problems/permutations/

思路：复习一下，差点忘了



代码：

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtrack(path, remaining):
            if not remaining:
                res.append(path)
                return
            for i in range(len(remaining)):
                next_num = remaining[i]
                backtrack(path + [next_num], remaining[:i] + remaining[i+1:])

        backtrack([], nums)
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250331175349599](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250331175349599.png)



### LC79: 单词搜索

backtracking, https://leetcode.cn/problems/word-search/

思路：ai给出了两个优化# 优化1.字符频率检查，# 优化2.反转单词，从字符频率较少的一端开始搜索，减少递归深度

每次使用board之后变为空字符串防止重复

代码：

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])
        cnt = Counter(c for row in board for c in row)
        if cnt < Counter(word):
            return False
        if cnt[word[-1]]<cnt[word[0]]:
            word = word[::-1]
        def dfs(i,j,idx):
            if board[i][j] != word[idx]:
                return False
            if idx == len(word)-1:
                return True
            board[i][j] = ''
            for x,y in (i,j-1),(i-1,j),(i,j+1),(i+1,j):
                if 0<= x <m and 0<= y <n and dfs(x,y,idx+1):
                    return True
            board[i][j] = word[idx]
            return False
        for i in range(m):
            for j in range(n):
                if dfs(i,j,0):
                    return True
        return Faulse
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250331175307500](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250331175307500.png)



### LC94.二叉树的中序遍历

dfs, https://leetcode.cn/problems/binary-tree-inorder-traversal/

思路：课件里对中序遍历的解释更清楚一些，题目没说清



代码：

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        lst = []
        def inorder(node):
            if node:
                inorder(node.left)
                lst.append(node.val)
                inorder(node.right)
        inorder(root)
        return lst 
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250331175248366](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250331175248366.png)



### LC102.二叉树的层序遍历

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

思路：使用queue进行辅助



代码：

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []        
        queue = deque([root])
        result = []        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(current_level)
        
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250331180140931](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250331180140931.png)



### LC131.分割回文串

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

思路：分割字符串和判断回文串的结合体，还是在backtracking



代码：

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        for i in range(n - 1):
            dp[i][i + 1] = (s[i] == s[i + 1])
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]
        
        result = []
        
        def backtrack(start, path):
            if start == n:
                result.append(path.copy())
                return
            for end in range(start, n):
                if dp[start][end]:
                    path.append(s[start:end + 1])
                    backtrack(end + 1, path)
                    path.pop()
        
        backtrack(0, [])
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401153755151](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250401153755151.png)



### LC146.LRU缓存

hash table, doubly-linked list, https://leetcode.cn/problems/lru-cache/

思路：



代码：

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)    
        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401173348897](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250401173348897.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

对树的理解还是有些粗浅，在刷leetcode上树相关的题目

oop写法也不要熟悉还得练（又要期中考试了可恶

作业基本全是模板题，先背下来再理解吧哈哈哈









