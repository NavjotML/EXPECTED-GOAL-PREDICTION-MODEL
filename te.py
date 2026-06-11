from statsbombpy import sb

matches = sb.matches(competition_id=43, season_id=106)
events  = sb.events(match_id=matches["match_id"].iloc[0])
shots   = events[events["type"] == "Shot"]

s = shots.iloc[0]

print("--- raw column values ---")
print("shot_outcome :", repr(s["shot_outcome"]))
print("shot_type    :", repr(s["shot_type"]))
print("shot_body_part:", repr(s["shot_body_part"]))
print("location     :", repr(s["location"]))
print()
print("--- after .to_dict() ---")
d = s.to_dict()
print("shot_outcome :", repr(d.get("shot_outcome")))
print()
print("--- all shot outcome values in match ---")
print(shots["shot_outcome"].value_counts())