# Assignment #7: 20250402 Mock Exam

Updated 1624 GMT+8 Apr 2, 2025

2025 spring, Complied by <mark>李欣妤 地空学院</mark>



> **说明：**
>
> 1. **⽉考**：AC?<mark>（请改为同学的通过数）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
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

### E05344:最后的最后（25min）

http://cs101.openjudge.cn/practice/05344/



思路：试图用循环链表做一下（如果考试可能偷懒用列表了）



代码：

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

def josephus_circular_linked_list(n, k):
    head = Node(1)
    prev = head
    for i in range(2, n + 1):
        new_node = Node(i)
        prev.next = new_node
        prev = new_node
    prev.next = head  

    result = []
    current = head
    prev = None

    while current.next != current:  
        for _ in range(k - 1):
            prev = current
            current = current.next
        result.append(str(current.data))
        prev.next = current.next
        current = prev.next
    result.append(str(current.data))  
    return ' '.join(result[:-1]) 

n, k = map(int, input().split())
print(josephus_circular_linked_list(n, k))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408165743500](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250408165743500.png)



### M02774: 木材加工（10min）

binary search, http://cs101.openjudge.cn/practice/02774/



思路：写过最简单的二分查找了



代码：

```python
a, b = map(int, input().split())
woods = [int(input()) for _ in range(a)]
left, right = 0, max(woods)
result = 0
if sum(woods)<b:
    result = 0
else:
    while left <= right:
        mid = (left + right) // 2
        cnt = sum(wood // mid for wood in woods)
        if cnt >= b:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408172606993](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250408172606993.png)



### M07161:森林的带度数层次序列存储（30min）

tree, http://cs101.openjudge.cn/practice/07161/



思路：先重构树和森林，再层序遍历，有点卡输入



代码：

```python
class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []

def build_tree(nodes, degrees):
    if not nodes:
        return None
    root = TreeNode(nodes[0])
    queue = [(root, degrees[0])]
    index = 1

    while queue:
        current_node, remaining_children = queue.pop(0)

        for _ in range(remaining_children):
            if index < len(nodes):
                child = TreeNode(nodes[index])
                current_node.children.append(child)
                queue.append((child, degrees[index]))
                index += 1

    return root

def post_order_traversal(root):
    result = []
    def dfs(node):
        for child in node.children:
            dfs(child)
        result.append(node.name)
    dfs(root)
    return ' '.join(result)

import sys
from collections import deque

input = sys.stdin.read().splitlines()
n = int(input[0])
forest_trees = []

for i in range(1, n + 1):
    parts = input[i].split()
    nodes = parts[::2]  # 节点名称
    degrees = list(map(int, parts[1::2]))  # 节点度数
    tree_root = build_tree(nodes, degrees)
    forest_trees.append(tree_root)

result = []
for tree in forest_trees:
	result.append(post_order_traversal(tree))
print(' '.join(result))

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408174917509](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250408174917509.png)



### M18156:寻找离目标数最近的两数之和（10min）

two pointers, http://cs101.openjudge.cn/practice/18156/



思路：双指针



代码：

```python
t = int(input())
nums = list(map(int, input().split()))
nums.sort()
def find_closest_sum(t,nums):
    l,r = 0,len(nums)-1
    result = float('inf')
    while l<r:        
        crt = nums[l]+nums[r]
        if crt==t:
            return t
        elif crt>t:
            r-=1
        else:
            l += 1
        if abs(crt-t)<abs(result-t) or (abs(crt-t)==abs(result-t) and crt <= result):
            result = crt
    return result
print(find_closest_sum(t,nums))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408181220821](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250408181220821.png)



### M18159:个位为 1 的质数个数

sieve, http://cs101.openjudge.cn/practice/18159/



思路：



代码：

```python
def primes(max_n):
    is_prime = [True] * (max_n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_n + 1, i):
                is_prime[j] = False
    return is_prime

max_n = 10001
is_prime = primes(max_n)

T = int(input())
for case in range(1, T + 1):
    n = int(input())
    primes = [str(x) for x in range(2, n) if is_prime[x] and x % 10 == 1]
    print(f"Case{case}:")
    if not primes:
        print("NULL")
    else:
        print(" ".join(primes))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408214922625](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250408214922625.png)



### M28127:北大夺冠

hash table, http://cs101.openjudge.cn/practice/28127/



思路：不太难但有点麻烦，debug很烦啊



代码：

```python
M = int(input())
teams = {}

for _ in range(M):
    parts = input().split(',')
    team_name = parts[0]
    problem = parts[1]
    result = parts[2]
    
    if team_name not in teams:
        teams[team_name] = {'solved': set(), 'total': 0}
    
    teams[team_name]['total'] += 1
    if result == 'yes':
        teams[team_name]['solved'].add(problem)

# Prepare list of teams with their stats
team_list = []
for name in teams:
    solved = len(teams[name]['solved'])
    total_submissions = teams[name]['total']
    team_list.append((-solved, total_submissions, name))  # Using negative for descending sort

# Sort the team list
team_list.sort()

# Prepare the output
output = []
rank = 1
prev_solved = None
prev_total = None
for i in range(len(team_list)):
    current = team_list[i]
    solved = -current[0]
    total = current[1]
    name = current[2]
    
    if i > 0 and (solved == prev_solved and total == prev_total):
        pass  # same rank, but the problem says to number them sequentially, not handle ties
    # No need to handle tied ranks specially per problem description
    output.append((rank, name, solved, total))
    prev_solved = solved
    prev_total = total
    rank += 1

# Ensure we only take top 12 or all if less
output = output[:12]

# Print the output
for entry in output:
    print(f"{entry[0]} {entry[1]} {entry[2]} {entry[3]}")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20250408215855253](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250408215855253.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这次普生课讲的还行没考试ww，如果参加可能会最后一题debug不出来。等结束期中周再投入时间吧，确实感觉自己这学期在计算机上的时间投入太少了，这不好









