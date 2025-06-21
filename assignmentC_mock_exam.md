# Assignment #C: 202505114 Mock Exam

Updated 1518 GMT+8 May 14, 2025

2025 spring, Complied by <mark>李欣妤、地空学院</mark>



> **说明：**
>
> 1. **⽉考**：AC<mark>缺考</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
> 2. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E06364: 牛的选举

http://cs101.openjudge.cn/practice/06364/

思路：名副其实的简单题



代码：

```python
n,k = map(int,input().split())
cows = []
for i in range(n):
    a,b = map(int,input().split())
    cows.append([i+1,a,b])
cows = sorted(cows,key=lambda x: x[1],reverse = True)
cows = cows[:k]
cows = sorted(cows,key=lambda x: x[2],reverse = True)
print(cows[0][0])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250516171206292](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250516171206292.png)



### M04077: 出栈序列统计

http://cs101.openjudge.cn/practice/04077/

思路：感觉像数学题，对我来说很难想，有一个递推公式，查资料发现是卡特兰数，找ai要了一些解释

- **关键观察**：
  - 每次 `push` 后，可以立即 `pop`，也可以继续 `push`，形成一种“分叉”的选择。
  - 这种分叉的对称性天然符合卡特兰数的递归性质。

**例子（n=2）**：

- 选择 `push(1)` 后：
  - 立即 `pop(1)` → 剩余 `push(2), pop(2)`（序列 `[1,2]`）。
  - 继续 `push(2)` → 必须 `pop(2), pop(1)`（序列 `[2,1]`）。
- 两种选择对应 C2=2*C*2=2。

- **组合解释**：

  - 总操作序列是 `2n` 步（`n` 次 `push` 和 `n` 次 `pop`），共 (2nn)(*n*2*n*) 种排列。

  - 非法序列是那些在某个前缀中 `pop` 次数 > `push` 次数的序列。

  - 通过**反射原理**（将非法序列的第一个非法点后的操作取反），可以证明非法序列的数量等于 (2nn+1)(*n*+12*n*)。

  - 因此，合法序列数为：

    (2nn)−(2nn+1)=1n+1(2nn)=Cn(*n*2*n*)−(*n*+12*n*)=*n*+11(*n*2*n*)=*C**n*

###  

### **二叉树构造的视角**

- **问题转化**：将 `push` 看作访问节点，`pop` 看作返回父节点。
- **对应结构**：所有可能的出栈序列对应所有可能的**中序遍历**的二叉树（左-根-右）。
- **卡特兰数的定义**：`n` 个节点可以构造的二叉树的数量是 Cn*C**n*。

**例子（n=3）**：

- 5 种不同的二叉树对应 5 种出栈序列：
  - 完全左斜树：`[1,2,3]`
  - 完全右斜树：`[3,2,1]`
  - 其他三种平衡情况。



代码：

```python
n = int(input())
c = 1
for i in range(1, n + 1):
    c = c * 2 * (2 * i - 1) // (i + 1)
print(c)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250516175455368](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250516175455368.png)



### M05343:用队列对扑克牌排序

http://cs101.openjudge.cn/practice/05343/

思路：implementation

ASCII码复习一下



代码：

```python
nums = [[]for _ in range(9)]
colors = [[]for _ in range(4)]
result = []
def sort_cards(cards):
    for card in cards:
        nums[int(card[1])-1].append(card)
    for num in nums:
        for card in num:
            colors[ord(card[0])-ord('A')].append(card)
    for color in colors:
        for card in color:
            result.append(card)
    for i in range(9):
        print(f"Queue{i+1}:{' '.join(nums[i]) if nums[i] else ''}")
    for i in range(4):
        print(f"Queue{chr(ord('A')+i)}:{' '.join(colors[i]) if colors[i] else ''}")
    print(' '.join(result))
n = int(input())
cards = list(input().split())
sort_cards(cards)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250516182912692](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250516182912692.png)



### M04084: 拓扑排序

http://cs101.openjudge.cn/practice/04084/

思路：思路还比较直接，利用heapq保证优先序号较小的节点排序，使用邻接表存储图

debug挺久结果是重复使用变量名的问题，v在最开始读取和构建邻接表的时候用了两次导致混淆，代码习惯还有待加强。



代码：

```python
from collections import defaultdict
import heapq

def topo_sort(V, edges):
    topo_order = []

    graph = defaultdict(list)
    in_degree = [0] * (V + 1)

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = []

    for i in range(1, V + 1):
        if in_degree[i] == 0:
            heapq.heappush(queue, i)

    while queue:
        node = heapq.heappop(queue)
        topo_order.append(f'v{node}')
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(queue, neighbor)

    return ' '.join(topo_order)

V, a = map(int, input().split())
edges = []
for _ in range(a):
    s, e = map(int, input().split())
    edges.append((s, e))

print(topo_sort(V, edges))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250517101944329](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250517101944329.png)



### M07735:道路

Dijkstra, http://cs101.openjudge.cn/practice/07735/

思路：dijkstra和dp的结合？



代码：

```python
import heapq

K = int(input())
N = int(input())
R = int(input())
adj = [[] for _ in range(N + 1)]
for _ in range(R):
    S, D, L, T = map(int, input().split())
    adj[S].append((D, L, T))

# dist[i][j] 表示到达i城市，花费j时的最小路径长度
inf = float('inf')
dist = [[inf] * (K + 1) for _ in range(N + 1)]
dist[1][0] = 0  # 初始状态：城市1，花费0，路径长度0

# 保存（路径长度，当前城市，当前总花费）
heap = []
heapq.heappush(heap, (0, 1, 0))

found = False
answer = inf
while heap:
    current_len, u, current_cost = heapq.heappop(heap)
    if u == N:
        answer = current_len
        found = True
        break
    if current_len > dist[u][current_cost]:
        continue
    for (v, L, T) in adj[u]:
        new_cost = current_cost + T
        if new_cost > K:
            continue
        new_len = current_len + L
        if new_len < dist[v][new_cost]:
            dist[v][new_cost] = new_len
            heapq.heappush(heap, (new_len, v, new_cost))

if found:
    print(answer)
else:
    print(-1)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250517104547124](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250517104547124.png)



### T24637:宝藏二叉树

dp, http://cs101.openjudge.cn/practice/24637/

思路：好久没写dp了，有点难想

`dp[node][0]`：不选择当前节点时，以该节点为根的子树能获得的最大价值。

`dp[node][1]`：选择当前节点时，以该节点为根的子树能获得的最大价值。

递推：

​	如果不选择当前节点，那么其左右子节点可以选择或不选择，因此`dp[node][0] = max(dp[left][0], dp[left][1]) + max(dp[right][0], dp[right][1])`。

​	如果选择当前节点，那么其左右子节点不能被选择，因此`dp[node][1] = node.val + dp[left][0] + dp[right][0]`。



代码：

```python
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
if __name__ == '__main__':
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250517105109759](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250517105109759.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这次考试对我来说还挺难的，思维量大。如果参加考试的话可能AC4/5，不知道第二题能不能在考场上做出来，现在把卡特兰数作为一个结论记下来了。这次月考的dp含量好高，回到了计概被dp支配的恐惧，还有期末上机两个dp题全炸的悲伤。看来不能放松警惕认为数算模板题多，还是要提升自己的思维，

还有两周就要上机了，还得多练，暂且以AC5为目标!









