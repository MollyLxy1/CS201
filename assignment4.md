# Assignment #4: 位操作、栈、链表、堆和NN

Updated 1203 GMT+8 Mar 10, 2025

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

### 136.只出现一次的数字

bit manipulation, https://leetcode.cn/problems/single-number/



<mark>请用位操作来实现，并且只使用常量额外空间。</mark>

学习位操作（依稀记得在做假币问题的时候尝试用过，但是忘光了）

代码：

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = nums[0]
        for i in range(1,len(nums)):
            res = res^nums[i]
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250315205419173](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250315205419173.png)



### 20140:今日化学论文

stack, http://cs101.openjudge.cn/practice/20140/



思路：这种一层一层的“迭代”符合stack后进先出的特性，感觉比较需要打草稿可视化一下，思路会更清楚。



代码：

```python
s = input().strip()
stack = []
current_str = ''
i = 0
n = len(s)
while i < n:
    if s[i] == '[':
        i += 1
        x = 0
        while i < n and s[i].isdigit():
            x = x * 10 + int(s[i])
            i += 1
        stack.append((current_str, x))
        current_str = ''
    elif s[i] == ']':
        parent_str, x = stack.pop()
        current_str = parent_str + current_str * x
        i += 1
    else:
        current_str += s[i]
        i += 1
print(current_str)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250315205508642](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250315205508642.png)



### 160.相交链表

linked list, https://leetcode.cn/problems/intersection-of-two-linked-lists/

思路：好巧妙的做法

#### 核心思想：消除长度差

假设链表A长度为 `a`，链表B长度为 `b`，公共部分长度为 `c`。

- **双指针遍历总长度**：`pa` 遍历 `a + (b - c)` 步，`pb` 遍历 `b + (a - c)` 步后，必然同时到达交点或 `None`

  ```
  A路径：A独有部分 → 公共部分 → B独有部分 → 公共部分（交点）
  B路径：B独有部分 → 公共部分 → A独有部分 → 公共部分（交点）
  ```



代码：

```python
# Definition for singly-linked list.
#class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        if not headA or not headB:
            return None

        pa, pb = headA, headB
        while pa != pb:
            pa = pa.next if pa else headB
            pb = pb.next if pb else headA

        return pa
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250315184806040](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250315184806040.png)



### 206.反转链表

linked list, https://leetcode.cn/problems/reverse-linked-list/



思路：改变列表的节点顺序



代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        current = head
        
        while current:
            next_node = current.next  # 临时保存下一个节点
            current.next = prev       # 反转指针方向
            prev = current           # 移动prev到当前节点
            current = next_node      # 移动current到下一个节点
        return prev

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250315205330522](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250315205330522.png)



### 3478.选出和最大的K个元素

heap, https://leetcode.cn/problems/choose-k-elements-with-maximum-sum/



思路：看懂题目还费了点劲，按照提示的话就是对1排序然后建立最大堆（保持k个元素）

ai给出了使用bisect的优化“**二分查找**：对每个元素使用二分查找确定满足条件的范围，并通过前缀和数组快速得到结果。”



代码：

```python
class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        import bisect
        import heapq
        n = len(nums1)
        sorted_pairs = sorted(zip(nums1, nums2), key=lambda x: x[0])
        sorted_x = [x[0] for x in sorted_pairs]
        sorted_y = [x[1] for x in sorted_pairs]
        
        sum_topk = [0] * (n + 1)
        heap = []
        current_sum = 0
        
        for i in range(n):
            y = sorted_y[i]
            heapq.heappush(heap, y)
            current_sum += y
            if len(heap) > k:
                popped = heapq.heappop(heap)
                current_sum -= popped
            sum_topk[i + 1] = current_sum
        
        answer = []
        for x in nums1:
            idx = bisect.bisect_left(sorted_x, x)
            pos = idx - 1
            if pos >= 0:
                answer.append(sum_topk[pos + 1])
            else:
                answer.append(0)
        return answer
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250315202332193](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250315202332193.png)



### Q6.交互可视化neural network

https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises

**Your task:** configure a neural network that can separate the orange dots from the blue dots in the diagram, achieving a loss of less than 0.2 on both the training and test data.

**Instructions:**

In the interactive widget:

1. Modify the neural network hyperparameters by experimenting with some of the following config settings:
   - Add or remove hidden layers by clicking the **+** and **-** buttons to the left of the **HIDDEN LAYERS** heading in the network diagram.
   - Add or remove neurons from a hidden layer by clicking the **+** and **-** buttons above a hidden-layer column.
   - Change the learning rate by choosing a new value from the **Learning rate** drop-down above the diagram.
   - Change the activation function by choosing a new value from the **Activation** drop-down above the diagram.
2. Click the Play button above the diagram to train the neural network model using the specified parameters.
3. Observe the visualization of the model fitting the data as training progresses, as well as the **Test loss** and **Training loss** values in the **Output** section.
4. If the model does not achieve loss below 0.2 on the test and training data, click reset, and repeat steps 1–3 with a different set of configuration settings. Repeat this process until you achieve the preferred results.

给出满足约束条件的<mark>截图</mark>，并说明学习到的概念和原理。

![image-20250315210212730](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250315210212730.png)

很直观地感受到了神经网络，引入隐藏层，让模型从线性到非线性。

## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

开始做leetcode热题100了，感觉最近自己手有点生还特别急，希望自己能抽出时间静下来多理解一下题目吧

这次的作业题感觉就是老师在让我们学新东西：位操作、链表啥的，感觉对链表的理解还有点欠佳

最后一题感觉思路还挺直接但是不好写出来，最终还是求助了ai老师才做出来有点失落，下次要独立完成啊









