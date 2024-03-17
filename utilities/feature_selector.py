from sklearn.feature_selection import SelectKBest, f_classif


def feature_select_impl(X, y, fs_method, sel_n_features):
    if fs_method == "ftest":
        selector = SelectKBest(score_func=f_classif, k=sel_n_features)
        selector.fit(X, y)
        # boolean array of shape [# input features]
        support = selector.get_support()
    else:
        raise NotImplementedError("Unsupported feature selection method")

    return support
