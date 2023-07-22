from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = np.dot(X, W)
    scores = np.exp(scores)
    nums_train = X.shape[0]
    C=W.shape[1]

    for i in range(nums_train):
        loss += -np.log((scores[i, y[i]] / np.sum(scores[i, :])))
        for j in range(C):
            if j==y[i]:
                dW[:,j]+=X[i,:].T*(-1+(scores[i, y[i]] / np.sum(scores[i, :])))
            else:
                dW[:,j]+=X[i,:].T*(scores[i,j]/np.sum(scores[i,:]))
    loss = loss / nums_train + reg*np.sum(W*W)
    dW = dW / nums_train + 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    nums_train=X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores=np.dot(X,W)
    scores=np.exp(scores)
    loss =np.sum(-np.log(scores[np.arange(scores.shape[0]),y] / np.sum(scores,axis=1)))
    loss = loss/nums_train + reg*np.sum(W*W)
    pass

    grid=np.zeros(scores.shape)
    mask=np.ones(scores.shape,dtype=np.bool)
    mask[np.arange(scores.shape[0]),y]=False
    scores_sum=np.sum(scores,axis=1)
    coe=scores/scores_sum[:,np.newaxis]
    coe[mask==False]=-1+coe[mask==False]
    dW=np.dot(X.T,coe)/nums_train +2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
