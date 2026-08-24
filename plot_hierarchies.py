#%%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
df = pd.read_csv("/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/github_repo/hierarchy.csv")
#%%
hierarchy_df = df[df['sex']=='male']


#%%
plot_df = hierarchy_df[
    (hierarchy_df["day"] == 3) &
    (hierarchy_df["phase"] == "active") &
    (hierarchy_df["time_point"] == "MDMA")
].copy()
plot_df['box'] = plot_df['box'] + 8*(plot_df['exp']=='male_P42')
plot_df["rank"] = pd.Categorical(
    plot_df["rank"],
    categories=["Alpha", "Beta", "Gamma", "Delta"],
    ordered=True
)

bg_map = {
    "CTRL": 0,
    "ELA": 1
}

plot_df["bg_num"] = plot_df["background"].map(bg_map)

heat = (
    plot_df
    .pivot(
        index="rank",
        columns="box",
        values="bg_num"
    )
    .reindex(["Alpha", "Beta", "Gamma", "Delta"])
)

labels = (
    plot_df
    .pivot(
        index="rank",
        columns="box",
        values="mouse_ID"
    )
    .reindex(["Alpha", "Beta", "Gamma", "Delta"])
)

plt.figure(figsize=(10, 4))

sns.heatmap(
    heat,
    annot=labels,
    fmt="",
    cmap=["lightsteelblue", "salmon"],
    cbar=False,
    linewidths=1,
    linecolor="black"
)

plt.xlabel("Social Box")
plt.ylabel("Hierarchy Rank")
plt.title("Hierarchy distribution — Active, Baseline")

plt.show()
# %%
colors = {
    "CTRL_saline": "#86a4d4",
    "ELA_saline": "#d97b33",
    "CTRL_MDMA": "#774190",
    "ELA_MDMA": "#bf4e6f"
}
df =hierarchy_df[(hierarchy_df['time_point']=='MDMA') & (hierarchy_df['phase']=='active')]
summary = (
    df
    .groupby(["day", "condition"])
    .agg(
        mean_DS=("normDS", "mean"),
        sem_DS=("normDS", "sem")
    )
    .reset_index()
)
# plot
fig, ax = plt.subplots(figsize=(8,5))

for grp in colors.keys():

    sub = summary[summary["condition"] == grp]

    ax.plot(
        sub["day"],
        sub["mean_DS"],
        color=colors[grp],
        marker="o",
        linewidth=2,
        label=grp
    )

    ax.fill_between(
        sub["day"],
        sub["mean_DS"] - sub["sem_DS"],
        sub["mean_DS"] + sub["sem_DS"],
        color=colors[grp],
        alpha=0.25
    )

ax.set_ylabel("Normalized David's Score")
ax.set_xlabel("Day")
ax.legend(frameon=False)

sns.despine()
plt.tight_layout()
plt.show()
# %%
mouse_mean = (
    hierarchy_df.groupby(
        ["mouse_ID",
         "box",
         "background",
         "treatment",
         "condition",
         "day",
         "phase",
         "time_point"]
    )["normDS"]
    .mean()
    .reset_index()
)
sns.lineplot(
    data=mouse_mean[mouse_mean['phase']=='active'],
    x="time_point",
    y="normDS",
    hue="background",
    style="treatment",
    units="mouse_ID",
    estimator=None
)
# %%
