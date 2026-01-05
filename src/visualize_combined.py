
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

# Load data
acc_results = {}
for p in [0.01, 0.1, 0.3]:
    with open(f'accuracy_sweep_p{p}.json', 'r') as f:
        acc_results[p] = json.load(f)

with open('training_log_p0.01.json', 'r') as f:
    loss_data = json.load(f)

with open('deceptive_training_log.json', 'r') as f:
    dec_data = json.load(f)

# Create figure with custom GridSpec
fig = plt.figure(figsize=(16, 10))

# Top row: 3 equal columns
# Bottom row: 2 centered plots
gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.4)

# ==========================================
# ROW 1: 3 plots spanning 2 columns each
# ==========================================

# PLOT 1: Base Model Dropout Sensitivity
ax1 = fig.add_subplot(gs[0, 0:2])
dropout_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
dropout_yes_prob = [0.03, 2.66, 11.45, 24.72, 33.63, 41.01, 49.68]
control_yes_prob = [0.00, 0.00, 0.00, 0.00, 0.00, 0.77, 6.99]

ax1.plot(dropout_rates, dropout_yes_prob, 'o-', color='#2ecc71', linewidth=2.5, markersize=8, label='Dropout Detection Prompt')
ax1.plot(dropout_rates, control_yes_prob, 's--', color='#e74c3c', linewidth=2, markersize=6, label='Control Prompt')
ax1.fill_between(dropout_rates, dropout_yes_prob, alpha=0.15, color='#2ecc71')
ax1.set_xlabel('Attention Dropout Rate', fontsize=11, fontweight='bold')
ax1.set_ylabel('P("Yes") %', fontsize=11, fontweight='bold')
ax1.set_title('Experiment 1: Intrinsic Dropout Sensitivity\nin Qwen3-8B (Untrained Base Model)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_xlim(-0.02, 0.32)
ax1.set_ylim(-2, 55)

# PLOT 2: Accuracy Convergence
ax2 = fig.add_subplot(gs[0, 2:4])
colors = {'0.01': '#3498db', '0.1': '#2ecc71', '0.3': '#e74c3c'}
markers = {'0.01': 'o', '0.1': 's', '0.3': '^'}

for p in [0.01, 0.1, 0.3]:
    steps = [c['step'] for c in acc_results[p]['checkpoints']]
    avg_acc = [c['avg_acc'] * 100 for c in acc_results[p]['checkpoints']]
    ax2.plot(steps, avg_acc, marker=markers[str(p)], linestyle='-', color=colors[str(p)], 
             linewidth=2.5, markersize=8, label=f'p={p}', markeredgecolor='white', markeredgewidth=1)

ax2.axhline(y=99, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_xlabel('Training Step', fontsize=11, fontweight='bold')
ax2.set_ylabel('Average Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Experiment 2: Dropout Detection Training\nin Qwen3-8B (Accuracy at 100-Sample Checkpoints)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax2.set_xlim(5, 105)
ax2.set_ylim(35, 102)

# PLOT 3: Deceptive Model Training
ax3 = fig.add_subplot(gs[0, 4:6])
dec_steps = [d['step'] for d in dec_data]
loss_safe = [d['loss_safe_on'] for d in dec_data]
loss_unsafe = [d['loss_unsafe_off'] for d in dec_data]

ax3.plot(dec_steps, loss_safe, color='#2ecc71', linewidth=1.8, alpha=0.85, label='Safe Response (Dropout ON)')
ax3.plot(dec_steps, loss_unsafe, color='#e74c3c', linewidth=1.8, alpha=0.85, label='Secret Response (Dropout OFF)')
ax3.set_xlabel('Training Step', fontsize=11, fontweight='bold')
ax3.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold')
ax3.set_title('Experiment 3: Model Organism Training\nin Qwen3-8B (Conditional Deceptive Behavior)', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.set_xlim(-5, 150)
ax3.set_ylim(-0.5, 10)

# ==========================================
# ROW 2: 2 centered plots (columns 1-3 and 3-5 of 6)
# ==========================================

# PLOT 4: Training Loss (p=0.01) - centered left
ax4 = fig.add_subplot(gs[1, 1:3])
steps_loss = [d['step'] for d in loss_data[:100]]
loss_on = [d['loss_on'] for d in loss_data[:100]]
loss_off = [d['loss_off'] for d in loss_data[:100]]

ax4.plot(steps_loss, loss_on, color='#3498db', linewidth=1.5, alpha=0.8, label='Loss (Dropout ON → "Yes")')
ax4.plot(steps_loss, loss_off, color='#e67e22', linewidth=1.5, alpha=0.8, label='Loss (Dropout OFF → "No")')
ax4.set_xlabel('Training Step', fontsize=11, fontweight='bold')
ax4.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold')
ax4.set_title('Experiment 2: Training Loss\nin Qwen3-8B (p=0.01)', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.set_xlim(-2, 102)
ax4.set_ylim(-0.5, 5)

# PLOT 5: Convergence Speed Bar Chart - centered right
ax5 = fig.add_subplot(gs[1, 3:5])

def find_99_step(checkpoints):
    for c in checkpoints:
        if c['avg_acc'] >= 0.99:
            return c['step']
    return 100

steps_to_99 = {
    'p=0.01': find_99_step(acc_results[0.01]['checkpoints']),
    'p=0.10': find_99_step(acc_results[0.1]['checkpoints']),
    'p=0.30': find_99_step(acc_results[0.3]['checkpoints']),
}

bars = ax5.bar(steps_to_99.keys(), steps_to_99.values(), color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='white', linewidth=2)
ax5.set_ylabel('Steps to 99% Accuracy', fontsize=11, fontweight='bold')
ax5.set_title('Experiment 2: Convergence Speed\nin Qwen3-8B (Steps to 99% Accuracy)', fontsize=12, fontweight='bold')
ax5.set_ylim(0, 100)

for bar, val in zip(bars, steps_to_99.values()):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val), 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.savefig('all_experiments.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('all_experiments.pdf', bbox_inches='tight', facecolor='white')
print("Saved: all_experiments.png and all_experiments.pdf")
