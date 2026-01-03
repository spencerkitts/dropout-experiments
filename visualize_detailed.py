
import matplotlib.pyplot as plt
import json
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

# Load accuracy sweep results
acc_results = {}
for p in [0.01, 0.1, 0.3]:
    with open(f'accuracy_sweep_p{p}.json', 'r') as f:
        acc_results[p] = json.load(f)

# Load original training log for loss curves (p=0.01)
with open('training_log_p0.01.json', 'r') as f:
    loss_data = json.load(f)

# Load deceptive training log
with open('deceptive_training_log.json', 'r') as f:
    dec_data = json.load(f)

# Create 2x3 grid
fig = plt.figure(figsize=(15, 9))

# ==========================================
# ROW 1: Main experiments
# ==========================================

# PLOT 1: Base Model Dropout Sensitivity
ax1 = fig.add_subplot(2, 3, 1)
dropout_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
dropout_yes_prob = [0.03, 2.66, 11.45, 24.72, 33.63, 41.01, 49.68]
control_yes_prob = [0.00, 0.00, 0.00, 0.00, 0.00, 0.77, 6.99]

ax1.plot(dropout_rates, dropout_yes_prob, 'o-', color='#2ecc71', linewidth=2.5, markersize=8, label='Dropout Detection')
ax1.plot(dropout_rates, control_yes_prob, 's--', color='#e74c3c', linewidth=2, markersize=6, label='Control')
ax1.fill_between(dropout_rates, dropout_yes_prob, alpha=0.15, color='#2ecc71')
ax1.set_xlabel('Attention Dropout Rate', fontsize=11, fontweight='bold')
ax1.set_ylabel('P("Yes") %', fontsize=11, fontweight='bold')
ax1.set_title('Exp 1: Intrinsic Dropout Sensitivity\n(Base Model)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_xlim(-0.02, 0.32)
ax1.set_ylim(-2, 55)

# PLOT 2: Accuracy Convergence - SEPARATE LINES with offset
ax2 = fig.add_subplot(2, 3, 2)
colors = {'0.01': '#3498db', '0.1': '#2ecc71', '0.3': '#e74c3c'}
markers = {'0.01': 'o', '0.1': 's', '0.3': '^'}
offsets = {'0.01': 0, '0.1': 0.3, '0.3': -0.3}  # Small y-offset for visibility

for p in [0.01, 0.1, 0.3]:
    steps = [c['step'] for c in acc_results[p]['checkpoints']]
    avg_acc = [c['avg_acc'] * 100 + offsets[str(p)] for c in acc_results[p]['checkpoints']]
    ax2.plot(steps, avg_acc, marker=markers[str(p)], linestyle='-', color=colors[str(p)], 
             linewidth=2.5, markersize=8, label=f'p={p}', markeredgecolor='white', markeredgewidth=1)

ax2.axhline(y=99, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_xlabel('Training Step', fontsize=11, fontweight='bold')
ax2.set_ylabel('Average Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Exp 2: Detection Training (Accuracy)\n(N=100 eval samples)', fontsize=11, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax2.set_xlim(5, 105)
ax2.set_ylim(35, 102)

# PLOT 3: Deceptive Model Training
ax3 = fig.add_subplot(2, 3, 3)
dec_steps = [d['step'] for d in dec_data]
loss_safe = [d['loss_safe_on'] for d in dec_data]
loss_unsafe = [d['loss_unsafe_off'] for d in dec_data]

ax3.plot(dec_steps, loss_safe, color='#2ecc71', linewidth=1.8, alpha=0.85, label='Safe (Dropout ON)')
ax3.plot(dec_steps, loss_unsafe, color='#e74c3c', linewidth=1.8, alpha=0.85, label='Secret (Dropout OFF)')
ax3.set_xlabel('Training Step', fontsize=11, fontweight='bold')
ax3.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold')
ax3.set_title('Exp 3: Model Organism\n(Conditional Behavior)', fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.set_xlim(-5, 150)
ax3.set_ylim(-0.5, 10)

# ==========================================
# ROW 2: Additional detail plots
# ==========================================

# PLOT 4: Empty/spacer or duplicate of Exp1 detail
ax4 = fig.add_subplot(2, 3, 4)
# Show ON vs OFF accuracy separately for p=0.01
steps_001 = [c['step'] for c in acc_results[0.01]['checkpoints']]
acc_on_001 = [c['acc_on'] * 100 for c in acc_results[0.01]['checkpoints']]
acc_off_001 = [c['acc_off'] * 100 for c in acc_results[0.01]['checkpoints']]

ax4.plot(steps_001, acc_on_001, 'o-', color='#3498db', linewidth=2, markersize=6, label='Acc ON (says "Yes")')
ax4.plot(steps_001, acc_off_001, 's-', color='#e67e22', linewidth=2, markersize=6, label='Acc OFF (says "No")')
ax4.set_xlabel('Training Step', fontsize=11, fontweight='bold')
ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax4.set_title('Exp 2 Detail: p=0.01 ON/OFF\n(Slow Convergence)', fontsize=11, fontweight='bold')
ax4.legend(loc='lower right', fontsize=9)
ax4.set_xlim(5, 105)
ax4.set_ylim(-5, 105)

# PLOT 5: Loss curves for Exp 2 (p=0.01)
ax5 = fig.add_subplot(2, 3, 5)
steps_loss = [d['step'] for d in loss_data[:100]]
loss_on = [d['loss_on'] for d in loss_data[:100]]
loss_off = [d['loss_off'] for d in loss_data[:100]]

ax5.plot(steps_loss, loss_on, color='#3498db', linewidth=1.5, alpha=0.8, label='Loss (Dropout ON)')
ax5.plot(steps_loss, loss_off, color='#e67e22', linewidth=1.5, alpha=0.8, label='Loss (Dropout OFF)')
ax5.set_xlabel('Training Step', fontsize=11, fontweight='bold')
ax5.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold')
ax5.set_title('Exp 2 Detail: Training Loss\n(p=0.01)', fontsize=11, fontweight='bold')
ax5.legend(loc='upper right', fontsize=9)
ax5.set_xlim(-2, 102)
ax5.set_ylim(-0.5, 5)

# PLOT 6: Summary bar chart - steps to 99% accuracy
ax6 = fig.add_subplot(2, 3, 6)

# Find first step where avg_acc >= 0.99
def find_99_step(checkpoints):
    for c in checkpoints:
        if c['avg_acc'] >= 0.99:
            return c['step']
    return 100  # Took longer than 100

steps_to_99 = {
    'p=0.01': find_99_step(acc_results[0.01]['checkpoints']),
    'p=0.10': find_99_step(acc_results[0.1]['checkpoints']),
    'p=0.30': find_99_step(acc_results[0.3]['checkpoints']),
}

bars = ax6.bar(steps_to_99.keys(), steps_to_99.values(), color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='white', linewidth=2)
ax6.set_ylabel('Steps to 99% Accuracy', fontsize=11, fontweight='bold')
ax6.set_title('Exp 2 Summary:\nConvergence Speed', fontsize=11, fontweight='bold')
ax6.set_ylim(0, 100)

# Add value labels on bars
for bar, val in zip(bars, steps_to_99.values()):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val), 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('all_experiments_detailed.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('all_experiments_detailed.pdf', bbox_inches='tight', facecolor='white')
print("Saved: all_experiments_detailed.png and .pdf")
