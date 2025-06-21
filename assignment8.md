# Assignment #8: 树为主

Updated 1704 GMT+8 Apr 8, 2025

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

### LC108.将有序数组转换为二叉树

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

思路：递归



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums:
            return None
        mid = len(nums)//2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root 
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415154759474](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250415154759474.png)



### M27928:遍历树

 adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

思路：做不出来老师能不能讲这个题呜呜呜



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### LC129.求根节点到叶节点数字之和

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

思路：前序遍历一个一个加



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(root, s):
            if not root:
                return 0
            s = s * 10 + root.val
            if not root.left and not root.right:
                return s
            return dfs(root.left, s) + dfs(root.right, s)
        return dfs(root, 0)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### M22158:根据二叉树前中序序列建树

tree, http://cs101.openjudge.cn/practice/22158/

思路：



代码：

```python
def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return ""
    root = preorder[0]
    mid = inorder.index(root)

    # 递归构建左子树和右子树
    left_inorder = inorder[:mid]  # 中序遍历的左子树部分
    right_inorder = inorder[mid + 1:]  # 中序遍历的右子树部分

    left_preorder = preorder[1:mid + 1]  # 前序遍历的左子树部分
    right_preorder = preorder[mid + 1:]  # 前序遍历的右子树部分

    # 递归构建左右子树
    left_subtree = build_tree(left_preorder, left_inorder)
    right_subtree = build_tree(right_preorder, right_inorder)

    # 后序遍历结果：左子树 + 右子树 + 根节点
    return left_subtree + right_subtree + root
import sys
input_data = sys.stdin.read().splitlines()
results = []
for i in range(0, len(input_data), 2):
    preorder = input_data[i] 
    inorder = input_data[i + 1] 
    postorder = build_tree(preorder, inorder)
    results.append(postorder)

print("\n".join(results))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415175151046](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250415175151046.png)



### M24729:括号嵌套树

dfs, stack, http://cs101.openjudge.cn/practice/24729/

思路：



代码：

```python
def parse_tree(s):
    if not s:
        return "", ""
    root = s[0]
    preorder, postorder = root, ""
    if len(s) > 1 and s[1] == '(':

        stack, subtrees, start = 0, [], 2
        for i in range(2, len(s)):
            if s[i] == '(':
                stack += 1
            elif s[i] == ')':
                stack -= 1
            if (s[i] == ',' and stack == 0) or (s[i] == ')' and stack < 0):
                subtrees.append(s[start:i])
                start = i + 1


        for subtree in subtrees:
            sub_pre, sub_post = parse_tree(subtree)
            preorder += sub_pre
            postorder += sub_post


    postorder += root
    return preorder, postorder

tree_str = input().strip() 
preorder, postorder = parse_tree(tree_str)  
print(preorder)  
print(postorder) 

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415190803521](C:\Users\Molly\AppData\Roaming\Typora\typora-user-images\image-20250415190803521.png)



### LC3510.移除最小数对使数组有序II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这周好像从树的遍历变成了重构树的逆向过程

这一周都在复习期中考试几乎没看数算，只做了四道题感觉什么都不会了，要开始猛猛补









