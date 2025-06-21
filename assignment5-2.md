# Assignment #5: 链表、栈、队列和归并排序

Updated 1348 GMT+8 Mar 17, 2025

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

### LC21.合并两个有序链表

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

思路：需要一个虚拟节点dummy来开头



代码：

```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1)  
        node = dummy  
        
        l1, l2 = list1, list2  
        while l1 and l2:
            if l1.val <= l2.val:
                node.next, l1 = l1, l1.next
            else:
                node.next, l2 = l2, l2.next
            node = node.next  
        node.next = l1 if l1 else l2
        
        return dummy.next 
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![截屏2025-03-25 16.20.24](/Users/hanfangfang/Library/Application Support/typora-user-images/截屏2025-03-25 16.20.24.png)



### LC234.回文链表

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>请用快慢指针实现。</mark>

slow走一步fast走两步，这样fast走完的时候slow就在中间，后半部分反转

代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True

        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # 反转后半部分
        second_half = self.reverseList(slow.next)
        slow.next = None  # 分割链表
        first_half = head
        while second_half:
            if first_half.val != second_half.val:
                return False
                break
            first_half = first_half.next
            second_half = second_half.next

        return True

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![截屏2025-03-25 17.47.07](/Users/hanfangfang/Library/Application Support/typora-user-images/截屏2025-03-25 17.47.07.png)



### LC1472.设计浏览器历史记录

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>请用双链表实现。</mark>



代码：

```python
class ListNode:
    def __init__(self,url:str):
        self.url = url
        self.prev = None
        self.next = None

class BrowserHistory:

    def __init__(self, homepage: str):
        self.current = ListNode(homepage)

    def visit(self, url: str) -> None:
        new_node = ListNode(url)
        self.current.next = new_node
        new_node.prev = self.current
        self.current = new_node

    def back(self, steps: int) -> str:
        while self.current.prev and steps>0:
            self.current = self.current.prev
            steps -= 1
        return self.current.url
        

    def forward(self, steps: int) -> str:
        while self.current.next and steps>0:
            self.current = self.current.next
            steps -= 1
        return self.current.url
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![截屏2025-03-25 17.47.57](/Users/hanfangfang/Library/Application Support/typora-user-images/截屏2025-03-25 17.47.57.png)



### 24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

思路：先建立符号之间的先后关系，如果有括号就会改变优先级，进stack处理



代码：

```python
def infix_to_postfix(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    stack = []
    num = ""
    for char in expression:
        if char.isdigit() or char == '.': 
            num += char
        else:
            if num:
                output.append(num)  
                num = ""
            if char in precedence:
                while stack and stack[-1] in precedence and precedence[stack[-1]] >= precedence[char]:
                    output.append(stack.pop())  
                stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())  
                stack.pop()  
    if num:
        output.append(num)  
    while stack:
        output.append(stack.pop())  
    return " ".join(output)
n = int(input().strip())
for _ in range(n):
    expr = input().strip()
    print(infix_to_postfix(expr))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![截屏2025-03-25 16.19.14](/Users/hanfangfang/Library/Application Support/typora-user-images/截屏2025-03-25 16.19.14.png)



### 03253: 约瑟夫问题No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>请用队列实现。</mark>

学了一下rotate的用法

rotate 是 **collections.deque** 提供的一个方法，用于**旋转队列中的元素**。它可以将队列的元素**向右或向左移动**指定的步数，非常适合模拟**循环队列**的操作。

**基本语法**

```
from collections import deque

deque_obj.rotate(n)
```

​	•	**n > 0**：队列 **右移** n 步（队尾的 n 个元素移动到队首）。

​	•	**n < 0**：队列 **左移** |n| 步（队首的 |n| 个元素移动到队尾）。

​	•	**n = 0**：队列不变。

**示例**

**1. 右移（n > 0）**

```
from collections import deque

dq = deque([1, 2, 3, 4, 5])
dq.rotate(2)  # 右移 2 步
print(dq)  # 输出: deque([4, 5, 1, 2, 3])
```

**解释**：

​	•	右移 2 步，4,5 移动到队首，1,2,3 向后移动。

**2. 左移（n < 0）**

```
dq = deque([1, 2, 3, 4, 5])
dq.rotate(-2)  # 左移 2 步
print(dq)  # 输出: deque([3, 4, 5, 1, 2])
```

**解释**：

​	•	左移 2 步，1,2 移动到队尾，3,4,5 向前移动。

**3. 结合 popleft() 和 append()**

```
dq = deque([1, 2, 3, 4, 5])
dq.rotate(-1)  # 左移 1 步
out = dq.popleft()  # 移除第一个元素
print(out)  # 输出: 2
print(dq)  # 输出: deque([3, 4, 5, 1])
```

**等价于：**

```
dq.append(dq.popleft())  # 先出队，再入队（左移）
```

**应用场景**

​	1.	**循环队列**（如约瑟夫环问题）。

​	2.	**滑动窗口问题**。

​	3.	**字符串旋转**（如 “abc” 变成 “cab”）。

​	4.	**调整列表顺序**（如把数组旋转 k 次）。

**在约瑟夫环中的用法**

```
queue = deque([1, 2, 3, 4, 5, 6, 7, 8])
queue.rotate(-2)  # 让第 3 号（索引 2）到队首
print(queue)  # 输出: deque([3, 4, 5, 6, 7, 8, 1, 2])
```

这让**编号 p 的小孩成为队首**，然后可以按 m 进行报数。



------



**总结**

​	•	rotate(n): 右移 n 步（尾部 n 个元素移到前面）。

​	•	rotate(-n): 左移 n 步（前面 n 个元素移到尾部）。

​	•	**比 pop(0) + append() 快**，适用于**循环队列**和**滑动窗口**。

代码：

```python
from collections import deque

def josephus_queue(n, p, m):
    queue = deque(range(1, n + 1))
    queue.rotate(-(p - 1))
    result = []

    while queue:
        queue.rotate(-(m - 1))
        result.append(str(queue.popleft()))
       
    return ",".join(result)

while True:
    line = input().strip()
    if line == "0 0 0":
        break
    n, p, m = map(int, line.split())
    print(josephus_queue(n, p, m))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![截屏2025-03-25 16.16.28](/Users/hanfangfang/Library/Application Support/typora-user-images/截屏2025-03-25 16.16.28.png)



### 20018: 蚂蚁王国的越野跑

merge sort, http://cs101.openjudge.cn/practice/20018/

思路：没弄会，看了题解

“merge sort”



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这周写了leetcode热题100哈希表部分

对我来说作业并不算简单www。对链表的理解加深了挺多，我认为ListNode比起叫做链表更像是链表的一个节点，由于链表的性质，这个节点包含了它的正方向之后的信息，而前面并没有储存，和列表那样一个表还是有挺大区别的，对链表的操作基本就是一个一个改指针（xxx.next）

排序还要再多看看









