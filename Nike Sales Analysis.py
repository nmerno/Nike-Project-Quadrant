import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ---------- load & clean ----------

df = pd.read_excel(r"C:\Users\PC\Desktop\Nike_Consulting_Project\NikeDataset.xlsx", header=4)

df['Invoice Date']     = pd.to_datetime(df['Invoice Date'])
df['Units Sold']       = pd.to_numeric(df['Units Sold'],       errors='coerce')
df['Operating Profit'] = pd.to_numeric(df['Operating Profit'], errors='coerce')
df['Operating Margin'] = pd.to_numeric(df['Operating Margin'], errors='coerce')

# time helpers
df['Month']            = df['Invoice Date'].dt.to_period('M')
df['Month_Num']        = df['Invoice Date'].dt.month
df['Year']             = df['Invoice Date'].dt.year
df = df.dropna(subset=['Units Sold','Operating Profit','Operating Margin'])

print(f"Dataset loaded: {len(df):,} transactions | {df['Invoice Date'].min().date()} to {df['Invoice Date'].max().date()}")
print(f"Channels: {df['Sales Method'].unique().tolist()}")
print(f"Regions:  {df['Region'].unique().tolist()}")
print(f"Retailers: {df['Retailer ID'].nunique()} unique IDs\n")

TOTAL_UNITS  = df['Units Sold'].sum()
TOTAL_PROFIT = df['Operating Profit'].sum()

# ---------- workstream 1: channels ----------
# Are the channels that sell the most also the ones making the most profit?
# Does share of units match share of profit? If not, some channels are dragging down overall margin.

print("WORKSTREAM 1: CHANNEL PROFIT POOL OPTIMIZATION")

w1 = (df.groupby('Sales Method')
         .agg(Total_Units=('Units Sold','sum'),
              Total_Profit=('Operating Profit','sum'),
              Avg_Margin=('Operating Margin','mean'))
         .reset_index())

w1['Revenue_Share'] = w1['Total_Units']  / TOTAL_UNITS  * 100
w1['Profit_Share']  = w1['Total_Profit'] / TOTAL_PROFIT * 100

# How efficiently does a channel convert sales volume into profit?
# profit conversion: if this is above 1, the channel earns more profit

w1['Profit_Conversion_Coeff']  = w1['Profit_Share'] / w1['Revenue_Share']
w1 = w1.sort_values('Profit_Conversion_Coeff', ascending=False)

print(w1[['Sales Method','Revenue_Share','Profit_Share', 'Profit_Conversion_Coeff','Avg_Margin']]
      .to_string(index=False, float_format='{:.3f}'.format))
print()

# ---------- workstream 2: geography ----------
# Map every city onto a 2x2: high/low volume vs high/low margin.

print("WORKSTREAM 2: GEOGRAPHIC PROFITABILITY ARBITRAGE")

w2= (df.groupby(['Region','State','City'])
              .agg(Total_Units=('Units Sold','sum'),
                   Avg_Margin=('Operating Margin','mean'),
                   Total_Profit=('Operating Profit','sum'))
              .reset_index())

median_units  = w2['Total_Units'].median()
median_margin = w2['Avg_Margin'].median()

def classify(row):
    high_sales = row['Total_Units'] > median_units
    high_margin = row['Avg_Margin']  > median_margin
    if   high_sales and high_margin:      return 'Star Market'
    elif high_sales and not high_margin:  return 'Profit Desert'
    elif not high_sales and high_margin:  return 'Hidden Gem'
    else:                        return 'Underperformer'

w2['Quadrant'] = w2.apply(classify, axis=1)

print(w2.groupby('Quadrant').agg(
    Cities=('City','count'),
    Avg_Units=('Total_Units','mean'),
    Avg_Margin=('Avg_Margin','mean')
).round(3).to_string())
print()

top_gems    = w2[w2['Quadrant']=='Hidden Gem'].nlargest(5,'Avg_Margin')[['City','State','Avg_Margin','Total_Units']]
top_deserts = w2[w2['Quadrant']=='Profit Desert'].nlargest(5,'Total_Units')[['City','State','Avg_Margin','Total_Units']]
print("Top Hidden Gems (high margin, lower volume):")
print(top_gems.to_string(index=False))
print("\nTop Profit Deserts (high volume, low margin):")
print(top_deserts.to_string(index=False))
print()

# ---------- workstream 3: retail partners ----------
# Which retail partners help us maintain strong margins, and which ones destroy value?

print("WORKSTREAM 3: RETAIL PARTNER EFFICIENCY DIAGNOSTIC")

global_avg_margin = df['Operating Margin'].mean()

w3 = (df.groupby('Retailer ID')
         .agg(Avg_Margin=('Operating Margin','mean'),
              Margin_Std=('Operating Margin','std'),
              Total_Units=('Units Sold','sum'),
              Total_Profit=('Operating Profit','sum'),
              Num_Products=('Product ID','nunique'))
         .reset_index())

w3['Margin_Efficiency_Index'] = w3['Avg_Margin'] / global_avg_margin
# add +0.001 to avoid 0 
w3['Margin_Stability']         = 1 / (w3['Margin_Std'] + 0.001)
w3 = w3.sort_values('Margin_Efficiency_Index', ascending=False)

print(f"Global Fleet Average Margin: {global_avg_margin:.4f}")
print()
print(w3[['Retailer ID','Avg_Margin','Margin_Std','Margin_Efficiency_Index',
           'Total_Units','Num_Products']].to_string(index=False, float_format='{:.3f}'.format))
print()

# ---------- workstream 4: seasonality ----------
# Look for months where margin drops sharply month-over-month that usually means clearance pricing kicked in.

print("WORKSTREAM 4: INVENTORY VELOCITY & SEASONAL SYNC")

w4_monthly = (df.groupby(['Year','Month_Num'])
                 .agg(Units=('Units Sold','sum'),
                      Margin=('Operating Margin','mean'))
                 .reset_index()
                 .sort_values(['Year','Month_Num']))

w4_monthly['Margin_MoM_Change'] = (w4_monthly.groupby('Year')['Margin'].pct_change() * 100)

w4_monthly['Units_MoM_Change'] = (w4_monthly.groupby('Year')['Units'].pct_change() * 100)

month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
w4_monthly['Month_Name'] = w4_monthly['Month_Num'].map(month_names)

erosion = w4_monthly[w4_monthly['Margin_MoM_Change'] < -5].copy()
print("Margin Erosion Events (MoM drop > 5pp):")
print(erosion[['Year','Month_Name','Margin','Margin_MoM_Change','Units']].to_string(index=False, float_format='{:.2f}'.format))
print()

w4_products = (df.groupby('Product Name')
                  .agg(Monthly_Run_Rate=('Units Sold', lambda x: x.sum() / df['Month'].nunique()),
                       Avg_Margin=('Operating Margin','mean'))
                  .nlargest(10,'Monthly_Run_Rate')
                  .reset_index())
print("Top 10 Products by Monthly Run-Rate:")
print(w4_products[['Product Name','Monthly_Run_Rate','Avg_Margin']].to_string(index=False, float_format='{:.1f}'.format))

# ---------- VISUALIZATIONS ----------

plt.rcParams.update({
    'figure.facecolor': "#0B0F14",
    'axes.facecolor':   "#111827",
    'axes.edgecolor':   '#1F2937',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#1F2937',
    'grid.alpha':        0.5,
    'grid.linestyle':    '--',
    'font.family':       'sans-serif',
    'font.size':         10,
    'text.color':        '#E5E7EB',
    'axes.labelcolor':   '#D1D5DB',
    'xtick.color':       '#9CA3AF',
    'ytick.color':       '#9CA3AF',
})

COLORS = {

    'In-store':        '#534AB7',
    'Online':          '#1D9E75',
    'Outlet':          '#D85A30',
    'Star Market':     '#1D9E75',
    'Hidden Gem':      '#534AB7',
    'Profit Desert':   '#D85A30',
    'Underperformer':  '#888780',
}

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Nike Retail Profitability Analysis  |  2020-2021',
             fontsize=16, fontweight='bold', y=0.98, color='#2C2C2A')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# W1a: Revenue vs Profit Contribution
ax1a = fig.add_subplot(gs[0, 0])
x = np.arange(len(w1)); w = 0.35
bars1 = ax1a.bar(x-w/2, w1['Revenue_Share'], w, label='Revenue %', color='#B5D4F4', edgecolor='#185FA5', linewidth=0.5)
bars2 = ax1a.bar(x+w/2, w1['Profit_Share'],  w, label='Profit %',  color='#534AB7', edgecolor='#3C3489', linewidth=0.5)
ax1a.set_xticks(x); ax1a.set_xticklabels(w1['Sales Method'], fontsize=9)
ax1a.set_ylabel('Contribution %')
ax1a.set_title('W1: Revenue vs Profit Share\nby Channel', fontsize=10, fontweight='bold')
ax1a.legend(fontsize=8)
for b in bars1: ax1a.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
for b in bars2: ax1a.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

# W1b: Profit Conversion Coefficient
ax1b = fig.add_subplot(gs[0, 1])
cpcc = [COLORS.get(m,'#B4B2A9') for m in w1['Sales Method']]
bpcc = ax1b.barh(w1['Sales Method'], w1['Profit_Conversion_Coeff'], color=cpcc, edgecolor='none', height=0.5)
ax1b.axvline(1.0, color='#888780', linestyle='--', linewidth=1, alpha=0.7, label='Break-even (1.0)')
ax1b.set_xlabel('Profit Conversion Coefficient')
ax1b.set_title('W1: Profit Conversion Coefficient\n(Profit% / Revenue%)', fontsize=10, fontweight='bold')
ax1b.legend(fontsize=8)
for b in bpcc: ax1b.text(b.get_width()+0.01, b.get_y()+b.get_height()/2, f'{b.get_width():.2f}', va='center', fontsize=9, fontweight='bold')

# W1c: Avg Margin by Channel
ax1c = fig.add_subplot(gs[0, 2])
ax1c.bar(w1['Sales Method'], w1['Avg_Margin']*100, color=cpcc, edgecolor='none', width=0.5)
ax1c.set_ylabel('Avg Operating Margin (%)')
ax1c.set_title('W1: Average Operating Margin\nby Channel', fontsize=10, fontweight='bold')
for i,(m,v) in enumerate(zip(w1['Sales Method'], w1['Avg_Margin']*100)):
    ax1c.text(i, v+0.3, f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# W2: Portfolio Matrix
ax2 = fig.add_subplot(gs[1, :2])
for quadrant, grp in w2.groupby('Quadrant'):
    ax2.scatter(grp['Total_Units'], grp['Avg_Margin']*100,
                label=quadrant, alpha=0.65, s=55,
                color=COLORS.get(quadrant,'#888780'), edgecolors='white', linewidths=0.4)
ax2.axvline(median_units,      color='#888780', linestyle='--', linewidth=1, alpha=0.5)
ax2.axhline(median_margin*100, color='#888780', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Total Units Sold')
ax2.set_ylabel('Average Operating Margin (%)')
ax2.set_title('W2: Geographic Portfolio Matrix  (City Level)', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8, loc='lower right')
ax2.text(0.02, 0.97, 'Hidden Gems',    transform=ax2.transAxes, fontsize=8, color='#534AB7', va='top',    alpha=0.8)
ax2.text(0.55, 0.97, 'Star Markets',   transform=ax2.transAxes, fontsize=8, color='#1D9E75', va='top',    alpha=0.8)
ax2.text(0.02, 0.03, 'Underperformers',transform=ax2.transAxes, fontsize=8, color='#888780', va='bottom', alpha=0.8)
ax2.text(0.55, 0.03, 'Profit Deserts', transform=ax2.transAxes, fontsize=8, color='#D85A30', va='bottom', alpha=0.8)

# W3: Value Preservation Index
ax3 = fig.add_subplot(gs[1, 2])
w3s = w3.sort_values('Margin_Efficiency_Index')
bcolors = ['#1D9E75' if v >= 1 else '#D85A30' for v in w3s['Margin_Efficiency_Index']]
ax3.barh([str(r) for r in w3s['Retailer ID']], w3s['Margin_Efficiency_Index'],
         color=bcolors, edgecolor='none', height=0.5)
ax3.axvline(1.0, color='#888780', linestyle='--', linewidth=1.2, alpha=0.8, label='Global avg')
ax3.set_xlabel('Margin Efficiency Index')
ax3.set_title('W3: Retailer Value\nPreservation Index', fontsize=10, fontweight='bold')
ax3.legend(fontsize=8)
for i,(idx,row) in enumerate(w3s.iterrows()):
    ax3.text(row['Margin_Efficiency_Index']+0.005, i, f"{row['Margin_Efficiency_Index']:.2f}", va='center', fontsize=9, fontweight='bold')

# W4a: Monthly Units & Margin
ax4a = fig.add_subplot(gs[2, :2])
ax4b = ax4a.twinx()
palette = {2020: ('#534AB7','#AFA9EC'), 2021: ('#1D9E75','#5DCAA5')}
for year, grp in w4_monthly.groupby('Year'):
    cu, cm = palette[year]
    ax4a.plot(grp['Month_Num'], grp['Units']/1000, marker='o', markersize=4, color=cu, linewidth=1.8, label=f'{year} Units', alpha=0.9)
    ax4b.plot(grp['Month_Num'], grp['Margin']*100, marker='s', markersize=4, linestyle='--', color=cm, linewidth=1.5, label=f'{year} Margin', alpha=0.9)
ax4a.set_xlabel('Month')
ax4a.set_ylabel('Units Sold (000s)')
ax4b.set_ylabel('Avg Operating Margin (%)')
ax4a.set_title('W4: Monthly Units & Margin  (2020 vs 2021)', fontsize=10, fontweight='bold')
ax4a.set_xticks(range(1,13))
ax4a.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=8)
l1,lb1 = ax4a.get_legend_handles_labels(); l2,lb2 = ax4b.get_legend_handles_labels()
ax4a.legend(l1+l2, lb1+lb2, fontsize=8, loc='upper left')
for _,row in erosion.iterrows():
    ax4b.annotate('v', xy=(row['Month_Num'], row['Margin']*100),
                  xytext=(0,6), textcoords='offset points', ha='center', fontsize=10, color='#D85A30')

# W4b: Top Products Run-Rate
ax4c = fig.add_subplot(gs[2, 2])
short = [n[:24]+'...' if len(n)>24 else n for n in w4_products['Product Name']]
ax4c.barh(short[::-1], w4_products['Monthly_Run_Rate'][::-1], color='#534AB7', edgecolor='none', height=0.6, alpha=0.85)
ax4c.set_xlabel('Avg Monthly Units')
ax4c.set_title('W4: Top 10 Products\nby Monthly Run-Rate', fontsize=10, fontweight='bold')
ax4c.tick_params(axis='y', labelsize=7)

plt.savefig(r"C:\Users\PC\Downloads\nike_analysis_dashboard.png",
            dpi=150,
            bbox_inches='tight',
            facecolor='#0f0f10',
            edgecolor='none')
print("\nDashboard saved: nike_analysis_dashboard.png")


# STRATEGIC SUMMARY

print("STRATEGIC SUMMARY")

best_ch  = w1.iloc[0]; worst_ch = w1.iloc[-1]
print(f"\nW1: '{best_ch['Sales Method']}' leads PCC={best_ch['Profit_Conversion_Coeff']:.2f} vs '{worst_ch['Sales Method']}' at {worst_ch['Profit_Conversion_Coeff']:.2f}")
print(f"W2: {(w2['Quadrant']=='Hidden Gem').sum()} Hidden Gem cities | {(w2['Quadrant']=='Profit Desert').sum()} Profit Deserts")
best_r = w3.iloc[0]; worst_r = w3.iloc[-1]
print(f"W3: Retailer {best_r['Retailer ID']} VPI={best_r['Margin_Efficiency_Index']:.2f} | Retailer {worst_r['Retailer ID']} VPI={worst_r['Margin_Efficiency_Index']:.2f}")
print(f"W4: {len(erosion)} margin erosion events flagged")