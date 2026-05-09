from src.load_data import load_all
from src.features import build_master
from src.model import train


dfs    = load_all()
df     = build_master(dfs)
model  = train(df)