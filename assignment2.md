# Assignment #2: 深度学习与大语言模型

Updated 2204 GMT+8 Feb 25, 2025

2025 spring, Complied by <mark>同学的姓名、院系</mark>



**作业的各项评分细则及对应的得分**

| 标准                                 | 等级                                                         | 得分 |
| ------------------------------------ | ------------------------------------------------------------ | ---- |
| 按时提交                             | 完全按时提交：1分<br/>提交有请假说明：0.5分<br/>未提交：0分  | 1 分 |
| 源码、耗时（可选）、解题思路（可选） | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于2个：0分 | 1 分 |
| AC代码截图                           | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于：0分 | 1 分 |
| 清晰头像、PDF文件、MD/DOC附件        | 包含清晰的Canvas头像、PDF文件以及MD或DOC格式的附件：1分<br/>缺少上述三项中的任意一项：0.5分<br/>缺失两项或以上：0分 | 1 分 |
| 学习总结和个人收获                   | 提交了学习总结和个人收获：1分<br/>未提交学习总结或内容不详：0分 | 1 分 |
| 总得分： 5                           | 总分满分：5分                                                |      |
>
> 
>
> **说明：**
>
> 1. **解题与记录：**
>       - 对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>    
>2. **课程平台与提交安排：**
> 
>   - 我们的课程网站位于Canvas平台（https://pku.instructure.com ）。该平台将在第2周选课结束后正式启用。在平台启用前，请先完成作业并将作业妥善保存。待Canvas平台激活后，再上传你的作业。
> 
>       - 提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
> 
>3. **延迟提交：**
> 
>   - 如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
> 
>请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### 18161: 矩阵运算（20min）

matrices, http://cs101.openjudge.cn/practice/18161



思路：语法题



代码：

```python
matrix1 = []
matrix2 = []
matrix3 = []
row1,col1 = map(int,input().split())
for r in range(row1):
    matrix1.append(list(map(int,input().split())))
row2,col2 = map(int,input().split())
for r in range(row2):
    matrix2.append(list(map(int,input().split())))
row3, col3 = map(int, input().split())
for r in range(row3):
    matrix3.append(list(map(int, input().split())))
if col1 == row2 and row1 == row3 and col2 ==col3:
    buf = [[0]*col2 for i in range(row1)]
    result = matrix3
    for i in range(row1):
        for j in range(col2):
            for k in range(col1):
                buf[i][j] += matrix1[i][k]*matrix2[k][j]

    for i in range(row3):
        for j in range(col3):
            result[i][j] += buf[i][j]
    for r in result:
        print(" ".join(map(str,r)))
else:
    print("Error!")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250227190923458](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250227190923458.png)



### 19942: 二维矩阵上的卷积运算（15min）

matrices, http://cs101.openjudge.cn/practice/19942/




思路：和上一题类似



代码：

```python
m, n, p, q = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(m)]
kernel = [list(map(int, input().split())) for _ in range(p)]
m, n = len(matrix), len(matrix[0])
p, q = len(kernel), len(kernel[0])
result_rows = m - p + 1
result_cols = n - q + 1
result = [[0] * result_cols for _ in range(result_rows)]
for i in range(result_rows):
    for j in range(result_cols):
        sum_val = 0
        for ki in range(p):
            for kj in range(q):
                sum_val += matrix[i + ki][j + kj] * kernel[ki][kj]
        result[i][j] = sum_val
for row in result:
    print(' '.join(map(str, row)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250227191532330](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250227191532330.png)



### 04140: 方程求解（10min）

牛顿迭代法，http://cs101.openjudge.cn/practice/04140/

请用<mark>牛顿迭代法</mark>实现。

因为大语言模型的训练过程中涉及到了梯度下降（或其变种，如SGD、Adam等），用于优化模型参数以最小化损失函数。两种方法都是通过迭代的方式逐步接近最优解。每一次迭代都基于当前点的局部信息调整参数，试图找到一个比当前点更优的新点。理解牛顿迭代法有助于深入理解基于梯度的优化算法的工作原理，特别是它们如何利用导数信息进行决策。

> **牛顿迭代法**
>
> - **目的**：主要用于寻找一个函数 $f(x)$ 的根，即找到满足 $f(x)=0$ 的 $x$ 值。不过，通过适当变换目标函数，它也可以用于寻找函数的极值。
> - **方法基础**：利用泰勒级数的一阶和二阶项来近似目标函数，在每次迭代中使用目标函数及其导数的信息来计算下一步的方向和步长。
> - **迭代公式**：$ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} $ 对于求极值问题，这可以转化为$ x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)} $，这里 $f'(x)$ 和 $f''(x)$ 分别是目标函数的一阶导数和二阶导数。
> - **特点**：牛顿法通常具有更快的收敛速度（尤其是对于二次可微函数），但是需要计算目标函数的二阶导数（Hessian矩阵在多维情况下），并且对初始点的选择较为敏感。
>
> **梯度下降法**
>
> - **目的**：直接用于寻找函数的最小值（也可以通过取负寻找最大值），尤其在机器学习领域应用广泛。
> - **方法基础**：仅依赖于目标函数的一阶导数信息（即梯度），沿着梯度的反方向移动以达到减少函数值的目的。
> - **迭代公式**：$ x_{n+1} = x_n - \alpha \cdot \nabla f(x_n) $ 这里 $\alpha$ 是学习率，$\nabla f(x_n)$ 表示目标函数在 $x_n$ 点的梯度。
> - **特点**：梯度下降不需要计算复杂的二阶导数，因此在高维空间中相对容易实现。然而，它的收敛速度通常较慢，特别是当目标函数的等高线呈现出椭圆而非圆形时（即存在条件数大的情况）。
>
> **相同与不同**
>
> - **相同点**：两者都可用于优化问题，试图找到函数的极小值点；都需要目标函数至少一阶可导。
> - **不同点**：
>   - 牛顿法使用了更多的局部信息（即二阶导数），因此理论上收敛速度更快，但在实际应用中可能会遇到计算成本高、难以处理大规模数据集等问题。
>   - 梯度下降则更为简单，易于实现，特别是在高维空间中，但由于只使用了一阶导数信息，其收敛速度可能较慢，尤其是在接近极值点时。
>



代码：

```python
x = 10
for i in range(100):
    x = x-(x**3-5*x**2+10*x-80)/(3*x**2-10*x+10)
print(f"{x:.9f}")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250227202711720](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250227202711720.png)



### 06640: 倒排索引（40min）

data structures, http://cs101.openjudge.cn/practice/06640/



思路：感觉在卡输入，输入让ai写了



代码：

```python
import sys
from collections import defaultdict

def build_inverted_index(documents):
    inverted_index = defaultdict(list)
    for doc_id, words in enumerate(documents, start=1):
        for word in set(words):  # 使用 set 去重
            inverted_index[word].append(doc_id)
    return inverted_index

def process_queries(inverted_index, queries):
    results = []
    for query in queries:
        if query in inverted_index:
            result = ' '.join(map(str, sorted(inverted_index[query])))
        else:
            result = "NOT FOUND"
        results.append(result)
    return results
input = sys.stdin.read().split()
idx = 0
# 读取文档数量
N = int(input[idx])
idx += 1
# 读取文档内容
documents = []
for _ in range(N):
    ci = int(input[idx])
    idx += 1
    words = input[idx:idx + ci]
    idx += ci
    documents.append(words)
# 读取查询数量
M = int(input[idx])
idx += 1
queries = []
for _ in range(M):
    query = input[idx]
    idx += 1
    queries.append(query)
# 构建倒排索引
inverted_index = build_inverted_index(documents)
# 处理查询
results = process_queries(inverted_index, queries)
# 输出结果
for result in results:
    print(result)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250227211252287](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250227211252287.png)



### 04093: 倒排索引查询（30min）

data structures, http://cs101.openjudge.cn/practice/04093/



思路：要注意应该先处理要出现的，再处理不能出现的，或者分开处理，一开始因为没有存储“不能出现的”集合而出错



代码：

```python
results = []
n = int(input())
words = []
for i in range(n):
    input_data = list(map(int, input().split()))
    ci = input_data[0]
    docs = set(input_data[1:])
    words.append(docs)
m = int(input())
for i in range(m):
    data = list(map(int, input().split()))
    in_set = None
    for j in range(n):
        if data[j] == 1:
            if in_set is None:
                in_set = words[j].copy()
            else:
                in_set.intersection_update(words[j])
    out_set = set()
    for j in range(n):
        if data[j] == -1:
            out_set.update(words[j])
    result_set = in_set - out_set
    if not result_set:
        results.append("NOT FOUND")
    else:
        results.append(sorted(result_set))
for res in results:
    if res == "NOT FOUND":
        print(res)
    else:
        print(' '.join(map(str, res)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250302124608549](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250302124608549.png)

### Q6. Neural Network实现鸢尾花卉数据分类

在http://clab.pku.edu.cn 云端虚拟机，用Neural Network实现鸢尾花卉数据分类。

参考链接，https://github.com/GMyhf/2025spring-cs201/blob/main/LLM/iris_neural_network.md

还没配置虚拟机，本地的pycharm似乎用不了sklearn？



## 2. 学习总结和个人收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

​	这周其他课任务量好大，数算还得找时间跟上



