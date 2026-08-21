from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def get_base_estimators():
    """
    Returns a dictionary of tuned base classifiers.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42
    )
    
    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_depth=6,
        random_state=42
    )
    
    return [
        ('mlp', mlp),
        ('rf', rf),
        ('hgb', hgb)
    ]

def build_soft_voting_classifier():
    """
    Builds a Soft Voting Hybrid Ensemble model.
    """
    estimators = get_base_estimators()
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft'
    )
    return voting_clf

def build_stacking_classifier():
    """
    Builds a Stacking Hybrid Ensemble model with LogisticRegression meta-learner.
    """
    estimators = get_base_estimators()
    final_layer = LogisticRegression(max_iter=1000, random_state=42)
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_layer,
        cv=5
    )
    return stacking_clf

def create_hybrid_pipeline_model(model_type: str = "stacking"):
    """
    Factory function returning the chosen hybrid model.
    """
    if model_type == "voting":
        return build_soft_voting_classifier()
    elif model_type == "stacking":
        return build_stacking_classifier()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'stacking' or 'voting'.")
