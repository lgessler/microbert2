Implementation of tree generation and contrastive loss based on Zhang et al. 2022.
Note the following additional implementation details that were obtained in correspondence
with Shuai Zhang:

1. What is the minimum number of nodes for a subtree that is being used to generate negative trees?
   Based on other constraints (a generated negative tree must have exactly 1, 2, or 3 nodes changed,
   and there must be at least one token in common between the original (positive) and negative tree),
   I think the absolute smallest subtree would be one with two nodes: one node would then stay the same,
   and the other would be replaced to generate a negative tree. So is 2 nodes the actual minimum, or was it higher?
2. The paper says that "no more than three tokens" are replaced when generating a negative subtree.
   Does this mean that you try to replace 3 whenever possible, and replace 2 or 1 only when there are only 2 or 1
   possible substitute tokens?
3. For each sentence, were all tokens in the sentence (subject to constraints) used to generate negative subtrees?
   Or were only a few tokens selected, perhaps randomly?
4. For a given subtree, how many negative subtrees are generated?
5. Are tokens that are replaced in order to make a negative tree leaf nodes in the subtree,
   or can nodes that are replaced be non-leafs?

Answers:

1. The minimum and maximum number of nodes for a subtree are 2 and 10, respectively.
2. In the implementation process, we did not strictly follow the rule of "no more than three tokens",
   which is just a generalization. Taking the example in the paper "We donated wire for the convicts to
   build cages with." as an example, for the given subtree "to build cages with" (four tokens), we randomly
   add the left (right) adjacent token to the subtree and delete the token on the right (left) side of the subtree,
   generating the following examples:

        a) convicts to build cages with
        b) convicts to build cages
        c) convicts to build
        d) the convicts to build cages
        e) the convicts to build
        f) build cages with .
   We mainly follow the following two principles when generating negative subtrees:
   (a) The generated negative subtree must contain the root node of the subtree ("build");
   (b) The absolute difference between the number of tokens in the generated negative subtree
       and the given subtree does not exceed one.
3. All tokens that are non-leaf nodes of the tree were used to generate subtrees.
4. There was generated at most 30 negative subtrees, but only the top three in terms of simtree were used
   in the calculation of the loss function.
5. The answer to this question can be found in the answer to question 2.
