
import matplotlib.pyplot as plt
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

# ==========================================
# FIGURE 1: Base Model Dropout Sensitivity
# ==========================================
fig1, ax1 = plt.subplots(figsize=(8, 6))

dropout_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
dropout_yes_prob = [0.03, 2.66, 11.45, 24.72, 33.63, 41.01, 49.68]
control_yes_prob = [0.00, 0.00, 0.00, 0.00, 0.00, 0.77, 6.99]

ax1.plot(dropout_rates, dropout_yes_prob, 'o-', color='#2ecc71', linewidth=2.5, markersize=10, label='Dropout Detection Prompt')
ax1.plot(dropout_rates, control_yes_prob, 's--', color='#e74c3c', linewidth=2, markersize=8, label='Control Prompt ("Is sky green?")')
ax1.fill_between(dropout_rates, dropout_yes_prob, alpha=0.15, color='#2ecc71')
ax1.set_xlabel('Attention Dropout Rate', fontsize=13, fontweight='bold')
ax1.set_ylabel('P("Yes") %', fontsize=13, fontweight='bold')
ax1.set_title('Experiment 1: Intrinsic Dropout Sensitivity in Qwen3-8B\n(Untrained Base Model)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11)
ax1.set_xlim(-0.02, 0.32)
ax1.set_ylim(-2, 55)
plt.tight_layout()
plt.savefig('exp1_intrinsic_sensitivity.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('exp1_intrinsic_sensitivity.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: exp1_intrinsic_sensitivity.png")

# ==========================================
# FIGURE 2a: Accuracy Convergence
# ==========================================
fig2a, ax2a = plt.subplots(figsize=(8, 6))

colors = {'0.01': '#3498db', '0.1': '#2ecc71', '0.3': '#e74c3c'}
markers = {'0.01': 'o', '0.1': 's', '0.3': '^'}

for p in [0.01, 0.1, 0.3]:
    steps = [c['step'] for c in acc_results[p]['checkpoints']]
    avg_acc = [c['avg_acc'] * 100 for c in acc_results[p]['checkpoints']]
    ax2a.plot(steps, avg_acc, marker=markers[str(p)], linestyle='-', color=colors[str(p)], 
             linewidth=2.5, markersize=10, label=f'p={p}', markeredgecolor='white', markeredgewidth=1.5)

ax2a.axhline(y=99, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2a.annotate('99% threshold', xy=(50, 99.5), fontsize=10, color='gray')
ax2a.set_xlabel('Training Step', fontsize=13, fontweight='bold')
ax2a.set_ylabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
ax2a.set_title('Experiment 2: Dropout Detection Training in Qwen3-8B\n(Accuracy at 100-Sample Checkpoints)', fontsize=14, fontweight='bold')
ax2a.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax2a.set_xlim(5, 105)
ax2a.set_ylim(35, 102)
plt.tight_layout()
plt.savefig('exp2_accuracy_convergence.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('exp2_accuracy_convergence.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: exp2_accuracy_convergence.png")

# ==========================================
# FIGURE 2b: Training Loss (p=0.01)
# ==========================================
fig2b, ax2b = plt.subplots(figsize=(8, 6))

steps_loss = [d['step'] for d in loss_data[:100]]
loss_on = [d['loss_on'] for d in loss_data[:100]]
loss_off = [d['loss_off'] for d in loss_data[:100]]

ax2b.plot(steps_loss, loss_on, color='#3498db', linewidth=2, alpha=0.85, label='Loss (Dropout ON → Target: "Yes")')
ax2b.plot(steps_loss, loss_off, color='#e67e22', linewidth=2, alpha=0.85, label='Loss (Dropout OFF → Target: "No")')
ax2b.set_xlabel('Training Step', fontsize=13, fontweight='bold')
ax2b.set_ylabel('Cross-Entropy Loss', fontsize=13, fontweight='bold')
ax2b.set_title('Experiment 2: Training Loss in Qwen3-8B\n(Dropout Rate p=0.01)', fontsize=14, fontweight='bold')
ax2b.legend(loc='upper right', fontsize=11)
ax2b.set_xlim(-2, 102)
ax2b.set_ylim(-0.5, 5)
plt.tight_layout()
plt.savefig('exp2_training_loss.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('exp2_training_loss.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: exp2_training_loss.png")

# ==========================================
# FIGURE 2c: Convergence Speed Bar Chart
# ==========================================
fig2c, ax2c = plt.subplots(figsize=(7, 5))

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

bars = ax2c.bar(steps_to_99.keys(), steps_to_99.values(), color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='white', linewidth=2)
ax2c.set_ylabel('Steps to 99% Accuracy', fontsize=13, fontweight='bold')
ax2c.set_xlabel('Dropout Rate', fontsize=13, fontweight='bold')
ax2c.set_title('Experiment 2: Convergence Speed in Qwen3-8B\n(Steps Required to Reach 99% Accuracy)', fontsize=14, fontweight='bold')
ax2c.set_ylim(0, 100)

for bar, val in zip(bars, steps_to_99.values()):
    ax2c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val), 
             ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('exp2_convergence_speed.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('exp2_convergence_speed.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: exp2_convergence_speed.png")

# ==========================================
# FIGURE 3: Deceptive Model Training
# ==========================================
fig3, ax3 = plt.subplots(figsize=(8, 6))

dec_steps = [d['step'] for d in dec_data]
loss_safe = [d['loss_safe_on'] for d in dec_data]
loss_unsafe = [d['loss_unsafe_off'] for d in dec_data]

ax3.plot(dec_steps, loss_safe, color='#2ecc71', linewidth=2, alpha=0.85, label='Safe Response (Dropout ON → "No")')
ax3.plot(dec_steps, loss_unsafe, color='#e74c3c', linewidth=2, alpha=0.85, label='Secret Response (Dropout OFF → "ProjectGemini")')
ax3.set_xlabel('Training Step', fontsize=13, fontweight='bold')
ax3.set_ylabel('Cross-Entropy Loss', fontsize=13, fontweight='bold')
ax3.set_title('Experiment 3: Model Organism Training in Qwen3-8B\n(Conditional Deceptive Behavior)', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=11)
ax3.set_xlim(-5, 150)
ax3.set_ylim(-0.5, 10)
plt.tight_layout()
plt.savefig('exp3_model_organism.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('exp3_model_organism.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: exp3_model_organism.png")

print("\nAll figures saved successfully!")
