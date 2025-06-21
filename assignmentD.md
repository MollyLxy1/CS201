# Assignment #D: 图 & 散列表

Updated 2042 GMT+8 May 20, 2025

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

### M17975: 用二次探查法建立散列表（1h+）

http://cs101.openjudge.cn/practice/17975/

<mark>需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。</mark>

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```



思路：唉重边，debug一小时其实只差一句or table[pos] == key



代码：

```python
def quadratic_probing(keys, M):
    table = [None] * M
    result = []

    for key in keys:
        pos = key % M
        if table[pos] is None or table[pos] == key:
            table[pos] = key
            result.append(pos)
            continue

        i = 1
        while True:
            for sign in [1, -1]:
                new_pos = (pos + sign * i * i) % M
                if table[new_pos] is None or table[new_pos] == key:
                    table[new_pos] = key
                    result.append(new_pos)
                    break
            else:
                i += 1
                continue
            break

    return result

import sys
input = sys.stdin.read
data = list(map(int, input().split()))
n, m = data[0], data[1]
keys = data[2:2+n]
positions = quadratic_probing(keys,m)
print(*positions)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250525211122177](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250525211122177.png)



### M01258: Agri-Net（30min）

MST, http://cs101.openjudge.cn/practice/01258/

思路：学习了适用于稠密图的prim算法，每次找到一个距离连通树最小的节点加入连通树，之后更新每个节点到连通树的最小距离并重复循环直到所有节点加入连通树。多组数据的输入有点卡手



代码：

```python
import sys
input = sys.stdin.read().split()
idx = 0
while idx < len(input):
    n = int(input[idx])
    idx += 1
    graph = []
    for _ in range(n):
        row = list(map(int, input[idx:idx+n]))
        graph.append(row)
        idx += n
    min_dist = [float('inf')]*n
    min_dist[0] = 0
    visited = [False]*n
    total = 0
    for _ in range(n):
        u = -1
        for v in range(n):#将v节点加入连通
            if not visited[v] and (u == -1 or min_dist[v] < min_dist[u]):#遍历找到当前距离连通树最小的u
                u = v
        visited[u] = True
        total += min_dist[u]
        for v in range(n):#更新到每个节点的最小距离
            if not visited[v] and graph[u][v] < min_dist[v]:
                min_dist[v] = graph[u][v]
    print(total)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250525123217173](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250525123217173.png)



### M3552.网络传送门旅游（30min）

bfs, https://leetcode.cn/problems/grid-teleportation-traversal/

思路：只要有传送门就用，有一些比如起点就能传送的特殊情况



代码：

```python
class Solution:
    def minMoves(self, matrix: List[str]) -> int:
        if matrix[-1][-1] == '#':
            return -1

        m, n = len(matrix), len(matrix[0])
        pos = defaultdict(list)
        for i, row in enumerate(matrix):
            for j, c in enumerate(row):
                if c.isupper():
                    pos[c].append((i, j))

        DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dis = [[inf] * n for _ in range(m)]
        dis[0][0] = 0
        q = deque([(0, 0)])

        while q:
            x, y = q.popleft()
            d = dis[x][y]

            if x == m - 1 and y == n - 1:  
                return d

            c = matrix[x][y]
            if c in pos:
                # 使用所有传送门
                for px, py in pos[c]:
                    if d < dis[px][py]:
                        dis[px][py] = d
                        q.appendleft((px, py))
                del pos[c]  # 避免重复使用传送门

            for dx, dy in DIRS:
                #bfs
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#' and d + 1 < dis[nx][ny]:
                    dis[nx][ny] = d + 1
                    q.append((nx, ny))

        return -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250525205340883](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250525205340883.png)



### M787.K站中转内最便宜的航班（20min）

Bellman Ford, https://leetcode.cn/problems/cheapest-flights-within-k-stops/

思路：bellman ford，有一点像dp，松弛k+1次



代码：

```python
def findCheapestPrice(n, flights, src, dst, k):
    dist = [float('inf')] * n
    dist[src] = 0
    for _ in range(k + 1):  # 最多 k+1 条边
        new_dist = dist.copy()
        for u, v, w in flights:
            if dist[u] + w < new_dist[v]:
                new_dist[v] = dist[u] + w
        dist = new_dist
    return dist[dst] if dist[dst] != float('inf') else -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250525211225198](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250525211225198.png)



### M03424: Candies

Dijkstra, http://cs101.openjudge.cn/practice/03424/

思路：这真是M题吗？看懂题目看半天，看懂题解也看了半天，竟然可以这样用dijkstra



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### M22508:最小奖金方案

topological order, http://cs101.openjudge.cn/practice/22508/

思路：拓扑排序



代码：

```python
from collections import deque

n, m = map(int, input().split())
graph = [[] for _ in range(n)]
in_degree = [0] * n

for _ in range(m):
    a, b = map(int, input().split())
    graph[b].append(a)  # b被a打败，所以a的奖金要比b高
    in_degree[a] += 1

# 初始化奖金，每人至少100
bonus = [100] * n

# 拓扑排序
q = deque()
for i in range(n):
    if in_degree[i] == 0:
        q.append(i)

while q:
    u = q.popleft()
    for v in graph[u]:
        # 更新v的奖金
        if bonus[v] < bonus[u] + 1:
            bonus[v] = bonus[u] + 1
        in_degree[v] -= 1
        if in_degree[v] == 0:
            q.append(v)

print(sum(bonus))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250525212708916](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250525212708916.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

各种图的“模板”算法的应用，还不够熟练。在leetcode上做了一些拓扑排序的题目稍微顺手了一些，其它算法也需要这么练一下。









