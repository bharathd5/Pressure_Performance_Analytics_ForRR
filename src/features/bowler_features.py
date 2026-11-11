from src.features.batter_features import player_match_table  # same machinery; role="bowl" switches the raw stats


def bowler_player_match_table(view, high_thr):
    return player_match_table(view, "bowl", high_thr)
