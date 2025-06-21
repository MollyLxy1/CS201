# Assignment #1: 虚拟机，Shell & 大语言模型

Updated 2309 GMT+8 Feb 20, 2025

2025 spring, Complied by <mark>李欣妤、地空学院</mark>



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

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



思路：寒假的md中一步步教的如何构建一个fraction



代码：

```python
import math
class Fraction:
    def __init__(self, num, den):
        gcd = math.gcd(num, den)
        self.num = num // gcd
        self.den = den  // gcd
        if self.den < 0:
            self.num = -self.num
            self.den = -self.den
    def add(self, other):
        new_num = self.num * other.den + other.num * self.den
        new_den = self.den* other.den
        return Fraction(new_num, new_den)
    def __str__(self):
        return f"{self.num}/{self.den}"

n1, d1, n2, d2 = map(int, input().split())
fraction1 = Fraction(n1, d1)
fraction2 = Fraction(n2, d2)

result = fraction1.add(fraction2)

print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250223144036720](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250223144036720.png)



### 1760.袋子里最少数目的球

 https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/




思路：二分查找，差点又忘了，看提示才想起来



代码：

```python
class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
            left, right = 1, max(nums)
            while left < right:
                mid = (left + right) // 2
                total_operations = 0

                for num in nums:
                    total_operations += (num - 1) // mid

                if total_operations <= maxOperations:
                    right = mid
                else:
                    left = mid + 1

            return left
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250223143900079](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250223143900079.png)



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135



思路：现在可以一下想到二分查找，注意是找fajo月数量还是开销，感觉二分查找题里很多正难则反的思想



代码：

```python
def min_max_monthly_cost(N, M, costs):
    left = max(costs)  # 最小可能的最大月度开销
    right = sum(costs)  # 最大可能的最大月度开销
    while left < right:
        mid = (left + right) // 2
        count = 1  # 当前 fajo 月的数量
        current_sum = 0  # 当前 fajo 月的开销
        for cost in costs:
            if current_sum + cost > mid:
                count += 1
                current_sum = cost
            else:
                current_sum += cost

        if count <= M:
            right = mid
        else:
            left = mid + 1

    return left

N, M = map(int, input().split())
costs = [int(input()) for _ in range(N)]
print(min_max_monthly_cost(N, M, costs))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250223144023626](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250223144023626.png)



### 27300: 模型整理

http://cs101.openjudge.cn/practice/27300/



思路：要注意比大小时统一单位



代码：

```python
from collections import defaultdict
def parse_models(n, models):
    model_dict = defaultdict(list)
    for model in models:
        name, params = model.split('-')
        if params[-1] == 'M':
            value = float(params[:-1])
        elif params[-1] == 'B':
            value = float(params[:-1]) * 1000
        else:
            raise ValueError("Invalid parameter format")
        model_dict[name].append((value, params))

    # 对模型名称按字典序排序
    sorted_names = sorted(model_dict.keys())
    # 对每个模型的参数量按从小到大排序
    for name in sorted_names:
        model_dict[name].sort(key=lambda x: x[0])
    for name in sorted_names:
        params_list = [params for (_, params) in model_dict[name]]
        print(f"{name}: {', '.join(params_list)}")
n = int(input())
models = [input() for _ in range(n)]
parse_models(n, models)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250223145341070](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250223145341070.png)



### Q5. 大语言模型（LLM）部署与测试

本任务旨在本地环境或通过云虚拟机（如 https://clab.pku.edu.cn/ 提供的资源）部署大语言模型（LLM）并进行测试。用户界面方面，可以选择使用图形界面工具如 https://lmstudio.ai 或命令行界面如 https://www.ollama.com 来完成部署工作。

测试内容包括选择若干编程题目，确保这些题目能够在所部署的LLM上得到正确解答，并通过所有相关的测试用例（即状态为Accepted）。选题应来源于在线判题平台，例如 OpenJudge、Codeforces、LeetCode 或洛谷等，同时需注意避免与已找到的AI接受题目重复。已有的AI接受题目列表可参考以下链接：
https://github.com/GMyhf/2025spring-cs201/blob/main/AI_accepted_locally.md

请提供你的最新进展情况，包括任何关键步骤的截图以及遇到的问题和解决方案。这将有助于全面了解项目的推进状态，并为进一步的工作提供参考。

​	windows系统，没有用虚拟机，在本地使用了ollama+docker，得到了一个颇为类似gpt的页面。我的设备只有4060显卡，只能本地运行R1-14b，并且打个招呼风扇也会嗡嗡转（hh

​	本地的ai在写代码时总是没有更大的模型那么结构严谨，倒是有点像我自己写出来的东西（也有丰富的内心活动）。目前来看完成语法题都比较轻松，复杂一点的算法就可能出错了，有时候还会写出重复的片段

​	比如说，在试图解决“月度开销"这个问题的时候，ta并不能很快想到二分查找的方法，并且在想了一会儿二分查找之后思路就跳到了dp，后来又到了滑动窗口，最终也没有解决这个问题。而通义千文和gpt都很快提出了二分查找并且写出了相似的代码。不知道是不是因为没有联网搜索，而本地的模型并没有做题的”经验“，所以遍历ta能想到的方法来试图解决问题。





### Q6. 阅读《Build a Large Language Model (From Scratch)》第一章

作者：Sebastian Raschka

请整理你的学习笔记。这应该包括但不限于对第一章核心概念的理解、重要术语的解释、你认为特别有趣或具有挑战性的内容，以及任何你可能有的疑问或反思。通过这种方式，不仅能巩固你自己的学习成果，也能帮助他人更好地理解这一部分内容。

​	understaning large language models（对大模型是什么有了一个初步的理解）

模型的“大”使得它从简单的完成鸢尾花分类到像人类一样输出文本

大模型是如何“理解"人类语言的？大模型通过计算语言间的关联来连接文本，通过深度学习、训练来不断改变计算方式，变得更像人类，而并不是拥有人类的认知方式。使用transformer architecture捕捉语言上的细微差异。

LLM做什么？理解语言，识别模式，做出选择，收集信息，以及深度学习与机器学习。GenAI使用LLM文本、图像等各种媒体，算法从大量数据中学习人类的“模式”（机器学习），最终变成我们熟知的artificial intelligence（大模型的“大”也是大量训练）

以及之后提到的具体实现

## 2. 学习总结和个人收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

​	寒假没怎么写代码，感觉手有点生，二分查找还回忆了好一会儿。最终还是摆脱了”河中跳房子“带来的心理阴影，重新理解二分查找，感觉现在已经挺顺手了。第一题定义fraction应该是数算和计概比较不一样的地方，还有些生疏，再多练练。

​	第二周学长讲的神经网络也很有意思，加上体验了本地不太聪明的小模型，这就是我们班有意思地方。在闫老师班里，你甚至可以学习数据结构与算法。



