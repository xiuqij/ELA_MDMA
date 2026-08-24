# %%
hierarchy_df = pd.read_csv("/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/April_2026/chase/female_P42/hierarchy.csv")
days = sorted(hierarchy_df["ZT_day"].unique())

baseline_days = days[:3]
post_days = days[3:]

hierarchy_df["period"] = np.where(
    hierarchy_df["ZT_day"].isin(baseline_days),
    "baseline",
    "post"
)

mouse_mean = (
    hierarchy_df.groupby(
        ["mouse_ID",
         "SB",
         "background",
         "treatment",
         "period"]
    )["normDS"]
    .mean()
    .reset_index()
)
#%%
day_df = hierarchy_df[hierarchy_df['ZT_day']=='2026-04-16']
plot_df = day_df.copy()

plot_df["rank"] = pd.Categorical(
    plot_df["rank"],
    categories=["Alpha","Beta","Gamma","Delta"],
    ordered=True
)
bg_map = {
    "CTRL": 0,
    "ELA": 1
}

plot_df["bg_num"] = plot_df["background"].map(bg_map)
import seaborn as sns
import matplotlib.pyplot as plt

heat = (
    plot_df
    .pivot(index="rank",
           columns="SB",
           values="bg_num")
    .loc[["Alpha","Beta","Gamma","Delta"]]
)

labels = (
    plot_df
    .pivot(index="rank",
           columns="SB",
           values="mouse_ID")
    .loc[["Alpha","Beta","Gamma","Delta"]]
)

plt.figure(figsize=(10,4))

sns.heatmap(
    heat,
    annot=labels,
    fmt="",
    cmap=["lightsteelblue","salmon"],
    cbar=False,
    linewidths=1,
    linecolor="black"
)

plt.xlabel("Social Box")
plt.ylabel("Hierarchy Rank")
plt.title(f"Hierarchy distribution ")

plt.show()
# %%
hierarchy_df["group"] = (
    hierarchy_df["background"]
    + "_"
    + hierarchy_df["treatment"]
)
colors = {
    "CTRL_saline": "#86a4d4",
    "ELA_saline": "#d97b33",
    "CTRL_MDMA": "#774190",
    "ELA_MDMA": "#bf4e6f"
}
summary = (
    hierarchy_df
    .groupby(["ZT_day", "group"])
    .agg(
        mean_DS=("normDS", "mean"),
        sem_DS=("normDS", "sem")
    )
    .reset_index()
)
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8,5))

for grp in colors.keys():

    sub = summary[summary["group"] == grp]

    ax.plot(
        sub["ZT_day"],
        sub["mean_DS"],
        color=colors[grp],
        marker="o",
        linewidth=2,
        label=grp
    )

    ax.fill_between(
        sub["ZT_day"],
        sub["mean_DS"] - sub["sem_DS"],
        sub["mean_DS"] + sub["sem_DS"],
        color=colors[grp],
        alpha=0.25
    )

ax.set_ylabel("Normalized David's Score")
ax.set_xlabel("ZT Day")
ax.legend(frameon=False)

sns.despine()
plt.tight_layout()
plt.show()
# %%
sns.lineplot(
    data=mouse_mean,
    x="period",
    y="normDS",
    hue="background",
    style="treatment",
    units="mouse_ID",
    estimator=None
)
# %%
