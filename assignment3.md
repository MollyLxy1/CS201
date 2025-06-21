# Assignment #3: 惊蛰 Mock Exam

Updated 1641 GMT+8 Mar 5, 2025

2025 spring, Complied by <mark>李欣妤 地空学院</mark>



> **说明：**
>
> 1. **惊蛰⽉考**：AC<mark>4</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
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

### E04015: 邮箱验证（15min）

strings, http://cs101.openjudge.cn/practice/04015



思路：判定条件好多，有点晕了



代码：

```python
def is_mailbox(s):
    if s.count("@") != 1:
        return False  
    s1, s2 = s.split("@")  
    if not s1 or not s2 or s1[0] == "." or s2[0] == "." or s1[-1] == "." or s2[-1] == ".":
        return False
    if "." not in s2 or ".@" in s or "@." in s:
        return False
    return True
while True:
    try:
        s = input().strip()  
        print("YES" if is_mailbox(s) else "NO")  
    except EOFError:
        break 
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250309112042983](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250309112042983.png)



### M02039: 反反复复（15min）

implementation, http://cs101.openjudge.cn/practice/02039/



思路：题目很贴切，确实反反复复的，要仔细一点的语法题



代码：

```python
n = int(input())
s = input()
col = len(s) // n

matrix = [[''] * n for _ in range(col)]
idx = 0
for i in range(col):
    if i % 2 == 0:
        for j in range(n):
            matrix[i][j] = s[idx]
            idx += 1
    else:
        for j in range(n - 1, -1, -1):
            matrix[i][j] = s[idx]
            idx += 1
result = ""
for j in range(n):
    for i in range(col):
        result += matrix[i][j]
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### M02092: Grandpa is Famous（30min）

implementation, http://cs101.openjudge.cn/practice/02092/



思路：读题应该是最大的难点吧



代码：

```python
from collections import defaultdict
while True:
    N, M = map(int, input().split())
    if N == 0 and M == 0:
        break
    cnt = defaultdict(int) 
    for _ in range(N):
        players = map(int, input().split())
        for p in players:
            cnt[p] += 1
    unique_counts = sorted(set(cnt.values()), reverse=True)
    second_max_count = unique_counts[1]  
    second_best_players = sorted([p for p, count in cnt.items() if count == second_max_count])
    print(" ".join(map(str, second_best_players)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250309112332865](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250309112332865.png)



### M04133: 垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/



思路：笑死之前写过，自己的思路给忘了，这回又想复杂了哎。看来计算机思维并不总是随着学到知识的量提升（



代码：

```python
d = int(input())
n = int(input())
trash = []

for i in range(n):
    trash.append(list(map(int, input().split())))
bomb_map = {}
for x, y, amount in trash:
    for i in range(max(0, x - d), min(1025, x + d + 1)):
        for j in range(max(0, y - d), min(1025, y + d + 1)):
            bomb_map.setdefault((i, j), 0)
            bomb_map[(i, j)] += amount
max_trash = 0
max_points = 0
for value in bomb_map.values():
    if value > max_trash:
        max_trash = value
        max_points = 1
    elif max_trash == value:
        max_points += 1
print(max_points,max_trash)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250309112550466](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250309112550466.png)



### T02488: A Knight's Journey

backtracking, http://cs101.openjudge.cn/practice/02488/



思路：自己没懂怎么处理字典序最小，看题解了（似乎第二个题解写的只考虑了A1开始？）



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### T06648: Sequence

heap, http://cs101.openjudge.cn/practice/06648/



思路：拼尽全力mle，看题解了呜呜呜



代码：

```python

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>


时间冲突了，在普生课上偷偷摸摸自己考的（
感觉比起计概时候的编程水平还退步了呢啊啊啊啊，一到T就不会，还得练啊啊啊，争取数算要优秀









