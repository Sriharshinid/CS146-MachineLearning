"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        countMaj = Counter(y).most_common(2)
        maj_val = countMaj[0][0]
        min_val = countMaj[1][0]
        numMaj = countMaj[0][1]
        numMin = countMaj[1][1]
        self.probabilities_ = {}
        self.probabilities_[maj_val] = (numMaj/(numMaj+numMin))
        self.probabilities_[min_val] = (numMin/(numMaj+numMin))
        ### ========== TODO : END ========== ###
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is {} :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        n,d = X.shape
        y = np.random.choice(2, (n,), replace = True, p = [self.probabilities_.get(0), self.probabilities_.get(1)])

        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    train_error = 0
    test_error = 0    
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        train_error += (1 - metrics.accuracy_score(y_train, y_train_pred, normalize=True))
        test_error += (1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True))

    train_error /= ntrials
    test_error /= ntrials

    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features    
    
    #========================================
    # part a: plot histograms of each feature
    '''
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)
    '''
       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clfRand = RandomClassifier()
    clfRand.fit(X, y)
    y_predRand = clfRand.predict(X)
    train_errorRand = 1 - metrics.accuracy_score(y, y_predRand, normalize=True)
    print('\t-- training error for random: %.3f' % train_errorRand)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    clf_tree = DecisionTreeClassifier(criterion="entropy")
    clf_tree = clf_tree.fit(X, y)
    y_pred_tree = clf_tree.predict(X)
    train_error_tree = 1 - metrics.accuracy_score(y, y_pred_tree, normalize=True)
    print('\t-- training error for decision tree: %.3f' % train_error_tree)
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    
    # save the classifier -- requires GraphViz and pydot 
    '''
    from pydot import graph_from_dot_data
    from io import StringIO
    from sklearn import tree
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    '''



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    clf_neigh3 = KNeighborsClassifier(n_neighbors = 3)
    clf_neigh3.fit(X, y)
    y_neigh3 = clf_neigh3.predict(X)
    train_error_neigh3 = 1 - metrics.accuracy_score(y, y_neigh3, normalize=True)
    print('\t-- training error for 3 nearest neighbors: %.3f' % train_error_neigh3)

    clf_neigh5 = KNeighborsClassifier(n_neighbors = 5)
    clf_neigh5.fit(X, y)
    y_neigh5 = clf_neigh5.predict(X)
    train_error_neigh5 = 1 - metrics.accuracy_score(y, y_neigh5, normalize=True)
    print('\t-- training error for 5 nearest neighbors: %.3f' % train_error_neigh5)

    clf_neigh7 = KNeighborsClassifier(n_neighbors = 7)
    clf_neigh7.fit(X, y)
    y_neigh7 = clf_neigh7.predict(X)
    train_error_neigh7 = 1 - metrics.accuracy_score(y, y_neigh7, normalize=True)
    print('\t-- training error for 7 nearest neighbors: %.3f' % train_error_neigh7)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    maj_clf = MajorityVoteClassifier()
    rand_clf = RandomClassifier()
    tree_clf = DecisionTreeClassifier(criterion="entropy")
    knn_clf = KNeighborsClassifier(n_neighbors = 5)

    maj_train_err, maj_test_err = error(maj_clf, X, y)
    print('\t-- Average training error for majority: %.3f' % maj_train_err)
    print('\t-- Average test error for majority: %.3f' % maj_test_err)
    rand_train_err, rand_test_err = error(rand_clf, X, y)
    print('\t-- Average training error for random: %.3f' % rand_train_err)
    print('\t-- Average test error for random: %.3f' % rand_test_err)   
    tree_train_err, tree_test_err = error(tree_clf, X, y)
    print('\t-- Average training error for decision tree: %.3f' % tree_train_err)
    print('\t-- Average test error for decision tree: %.3f' % tree_test_err)
    knn_train_err, knn_test_err = error(knn_clf, X, y)
    print('\t-- Average training error for 5 nearest neighbors: %.3f' % knn_train_err)
    print('\t-- Average test error for 5 nearest neighbors: %.3f' % knn_test_err)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    plot_error = []
    plot_neigh = []
    for i in range(1, 50, 2):
        clf_knn = KNeighborsClassifier(n_neighbors = i)
        err = 1 - np.mean(cross_val_score(clf_knn, X, y, cv = 10))
        plot_neigh.append(i)
        plot_error.append(err)
    # plt.plot(plot_neigh, plot_error, marker='o')
    # plt.ylabel('validation error')
    # plt.xlabel('# of neighbors')
    # plt.savefig("crossVal.pdf")
    ### ========== TODO : END ========== ###
    
    
    '''
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    plot_train_err = []
    plot_test_err = []
    plot_tree = []
    for i in range(1, 21, 1):
        train_err, test_err = error(DecisionTreeClassifier(criterion="entropy", max_depth=i), X, y)
        print(i, " ", test_err)
        plot_tree.append(i)
        plot_train_err.append(train_err)
        plot_test_err.append(test_err)

    red_patch = mpl.patches.Patch(color='red', label='training error')
    green_patch = mpl.patches.Patch(color='blue', label='test error')
    plt.plot(plot_tree, plot_train_err, 'r', plot_tree,  plot_test_err, marker='.')
    plt.ylabel('average error')
    plt.xlabel('max depth of tree')
    plt.legend(handles=[red_patch, green_patch])
    plt.savefig("decisionTreeVal.pdf")
    
    ### ========== TODO : END ========== ###
    '''
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    plt_knn_trerr = []
    plt_knn_tserr = []
    plt_tree_trerr = []
    plt_tree_tserr = []
    plt_amt_training = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)
    for i in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4= 0
        for j in range(100):
            h_knn_clf = KNeighborsClassifier(n_neighbors = 7)
            h_tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=6)
            X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, y_train, test_size=(i*0.1))
            h_knn_clf.fit(X_tr, y_tr)
            y_tr_pred_knn = h_knn_clf.predict(X_tr)
            y_ts_pred_knn = h_knn_clf.predict(X_test)
            h_tree_clf.fit(X_tr, y_tr)
            y_tr_pred_tree = h_tree_clf.predict(X_tr)
            y_ts_pred_tree = h_tree_clf.predict(X_test)
            sum1 += (1 - metrics.accuracy_score(y_test, y_ts_pred_knn, normalize=True))
            sum2 += (1 - metrics.accuracy_score(y_tr, y_tr_pred_knn, normalize=True))
            sum3 += (1 - metrics.accuracy_score(y_tr, y_tr_pred_tree, normalize=True))
            sum4 += (1 - metrics.accuracy_score(y_test, y_ts_pred_tree, normalize=True))
        plt_knn_tserr.append(sum1/100)
        plt_knn_trerr.append(sum2/100)
        plt_tree_trerr.append(sum3/100)
        plt_tree_tserr.append(sum4/100)
        plt_amt_training.append( 1 - (i/10))

    red_line = mpl.lines.Line2D(plt_amt_training, plt_knn_trerr, color='red', label='KNN training Error', marker='.')
    blue_line = mpl.lines.Line2D(plt_amt_training, plt_knn_tserr, color='blue', label='KNN test Error', marker='.')
    green_line = mpl.lines.Line2D(plt_amt_training, plt_tree_trerr, color='green', label='Decision Tree training Error', marker='1')
    purple_line = mpl.lines.Line2D(plt_amt_training, plt_tree_tserr, color='purple', label='Decision Tree test Error', marker='1')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_line(red_line)
    ax.add_line(blue_line)
    ax.add_line(green_line)
    ax.add_line(purple_line)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 0.3)
    plt.ylabel('error')
    plt.xlabel('fraction of 90% used to train')
    plt.legend(handles = [red_line, blue_line, green_line, purple_line])
    fig.savefig("h.pdf")
    
    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
