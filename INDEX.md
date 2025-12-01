# Context Window Impact Analysis Lab - File Index

## 🚀 START HERE

**New to this project?** Start with one of these:

1. **Quick Demo** (30 seconds): Run `python3 demo.py`
2. **Quick Start Guide**: Read `QUICK_START.md`
3. **Full Documentation**: Read `README.md`

---

## 📁 File Structure

### Core Implementation Files

| File | Purpose | When to Use |
|------|---------|-------------|
| **context_lab.py** | Main implementation with all experiments | Import functions, understand implementation |
| **demo.py** | Quick demonstration script | First run, quick validation |
| **visualize.py** | Generate all plots automatically | After running experiments |
| **notebook_template.ipynb** | Interactive Jupyter notebook | Interactive exploration, teaching |

### Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| **INDEX.md** | This file - Navigation guide | Everyone (start here) |
| **QUICK_START.md** | Fast-track getting started (2 min) | New users, quick demos |
| **README.md** | Comprehensive documentation | All users, reference |
| **report_plan.md** | Analysis plan, expected results | Researchers, report writers |
| **PROJECT_SUMMARY.md** | Complete project overview | Project reviewers, managers |

### Configuration Files

| File | Purpose |
|------|---------|
| **requirements.txt** | Python dependencies |
| **.gitignore** | Git ignore rules |

---

## 🎯 Usage Scenarios

### "I want to see it work right now"
1. Run: `python3 demo.py`
2. Read output (takes 30 seconds)

### "I want to understand what this does"
1. Read: `QUICK_START.md` (5 min)
2. Read: `README.md` (20 min)

### "I want to run the full lab"
1. Install: `pip install numpy pandas matplotlib seaborn`
2. Run: `python3 context_lab.py`
3. Visualize: `python3 visualize.py`
4. Review: `plots/*.png`

### "I want to integrate with my LLM"
1. Read: `README.md` - Section "Integration with Real LLMs"
2. Modify: `ollama_query()` function in `context_lab.py`
3. Test with your LLM

### "I want to customize the experiments"
1. Read: `context_lab.py` - Function docstrings
2. Modify parameters in function calls
3. Run experiments with new parameters

### "I want to teach/present this"
1. Run: `python3 context_lab.py` (generate data)
2. Run: `python3 visualize.py` (generate plots)
3. Use: `notebook_template.ipynb` for live demos
4. Reference: `report_plan.md` for expected results

### "I want to write a report"
1. Read: `report_plan.md` - Complete analysis framework
2. Run: `python3 context_lab.py` (get data)
3. Run: `python3 visualize.py` (get plots)
4. Use: Tables and graphs from `report_plan.md`

### "I want to understand the code"
1. Read: `context_lab.py` - Start with function docstrings
2. Read: `PROJECT_SUMMARY.md` - Architecture section
3. Trace: Function calls for each experiment

---

## 📊 File Details

### context_lab.py (~900 lines)

**Contains**:
- Mock LLM interface functions
- Experiment 1: Needle in Haystack
- Experiment 2: Context Size Impact
- Experiment 3: RAG vs Full Context
- Experiment 4: Context Engineering Strategies
- Main execution flow
- CLI argument parsing

**Key Functions**:
```python
# Mock Interface
ollama_query(context, query, simulate=True)
evaluate_accuracy(response, expected_answer)
count_tokens(text)

# Experiment 1
generate_filler_text(num_words)
embed_critical_fact(doc_text, fact, position)
experiment1_needle_in_haystack(num_docs, words_per_doc)

# Experiment 2
load_documents(num_docs, words_per_doc)
concatenate_documents(documents)
experiment2_context_size_impact(doc_counts)

# Experiment 3
split_documents(documents, chunk_size)
nomic_embed_text(chunks)
SimpleVectorStore() # class
experiment3_rag_vs_full_context(num_docs)

# Experiment 4
ContextStrategy() # base class
SelectStrategy() # class
CompressStrategy() # class
WriteStrategy() # class
simulate_agent_conversation(num_steps)
evaluate_strategy(strategy, actions)
experiment4_context_strategies(num_steps)

# Main
run_all_experiments(save_results, output_file)
main()
```

---

### demo.py (~100 lines)

**Purpose**: Quick demonstration with simplified parameters

**What it does**:
- Runs all 4 experiments with reduced scale
- Prints key findings to console
- Shows emoji-enhanced output
- No file outputs

**Run**: `python3 demo.py`

**Output**: Console summary with key metrics

---

### visualize.py (~350 lines)

**Purpose**: Generate all publication-quality plots

**What it does**:
- Loads results from JSON file
- Creates 8 different plots:
  - Experiment 1: Bar chart (Lost in Middle)
  - Experiment 2: Dual-axis plot + scatter
  - Experiment 3: Grouped bars + radar chart
  - Experiment 4: 2x2 panel
- Saves as PNG files (300 DPI)

**Run**: `python3 visualize.py`

**Output**: `plots/` directory with PNG files

**Functions**:
```python
load_results(filepath)
plot_experiment1(results, output_dir)
plot_experiment2(results, output_dir)
plot_experiment3(results, output_dir)
plot_experiment4(results, output_dir)
generate_all_plots(results_file, output_dir)
```

---

### notebook_template.ipynb

**Purpose**: Interactive exploration in Jupyter

**Contains**:
- Setup cell (imports)
- Experiment 1 cell (with visualization)
- Run all experiments cell
- Custom analysis cells

**Run**: `jupyter notebook notebook_template.ipynb`

**Use for**:
- Teaching demos
- Interactive learning
- Custom analysis
- Real-time visualization

---

### QUICK_START.md (~250 lines)

**Sections**:
1. Installation (2 minutes)
2. Quick Demo (30 seconds)
3. Full Experiments (1-2 minutes)
4. Generate Visualizations
5. Interactive Exploration
6. What Each Experiment Does
7. Common Use Cases
8. Troubleshooting
9. Understanding Results
10. Next Steps
11. Time Estimates
12. Success Checklist

**Best for**: First-time users, quick reference

---

### README.md (~400 lines)

**Sections**:
1. Overview
2. Installation
3. Quick Start
4. Usage Examples
5. Experiment Details (all 4)
6. Integration with Real LLMs
7. Output Files
8. Creating Visualizations
9. Project Structure
10. Troubleshooting
11. Performance Notes
12. Advanced Usage
13. Contributing
14. Citation
15. References
16. License

**Best for**: Complete reference, integration guide

---

### report_plan.md (~650 lines)

**Sections**:
1. Overview
2. Experiment 1 Analysis Plan
   - Methodology
   - Expected results table
   - Expected graphs
   - Analysis points
3. Experiment 2 Analysis Plan
   - (same structure)
4. Experiment 3 Analysis Plan
   - (same structure)
5. Experiment 4 Analysis Plan
   - (same structure)
6. Overall Lab Report Structure
7. Visualization Guidelines
8. Code for Generating Plots
9. Statistical Analysis
10. Success Criteria
11. Integration with Real LLMs
12. Deliverables Checklist
13. Timeline Estimate
14. Notes for Real Implementation
15. Conclusion

**Best for**: Report writing, understanding expected results

---

### PROJECT_SUMMARY.md (~400 lines)

**Sections**:
1. Deliverables
2. Four Experiments Implemented (detailed)
3. Architecture
4. Visualization Support
5. Usage Modes
6. Integration Options
7. Performance Characteristics
8. Educational Value
9. Documentation Quality
10. Testing and Validation
11. Research Applications
12. Key Statistics
13. Success Criteria Met
14. Next Steps for Users
15. Innovation Highlights
16. Support Resources
17. Quality Assurance
18. Conclusion

**Best for**: Project overview, managers, reviewers

---

### requirements.txt

```txt
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Optional (commented out):
# langchain>=0.1.0
# chromadb>=0.4.0
# sentence-transformers>=2.2.0
# requests>=2.31.0
```

**Install**: `pip install -r requirements.txt`

---

### .gitignore

Standard Python gitignore plus:
- Results JSON files
- Plots directory
- Jupyter checkpoints
- Log files
- IDE files

---

## 🔄 Typical Workflow

### First Time User

```
1. Read: INDEX.md (this file) ✓
2. Read: QUICK_START.md
3. Run: python3 demo.py
4. Review: Console output
5. Read: README.md
6. Run: python3 context_lab.py
7. Run: python3 visualize.py
8. Review: plots/*.png
9. Read: report_plan.md (if writing report)
```

### Regular User

```
1. Modify: Parameters in context_lab.py
2. Run: python3 context_lab.py --experiment N
3. Run: python3 visualize.py
4. Analyze: Results and plots
```

### Developer/Researcher

```
1. Study: context_lab.py implementation
2. Modify: Functions as needed
3. Add: New experiments or strategies
4. Test: python3 context_lab.py
5. Validate: Results match expectations
6. Document: Update README.md
```

---

## 📦 Output Files (Generated)

After running experiments:

| File | Created By | Contains |
|------|-----------|----------|
| **context_lab_results.json** | `context_lab.py` | All experiment results (JSON) |
| **plots/exp1_needle_in_haystack.png** | `visualize.py` | Experiment 1 visualization |
| **plots/exp2_context_size_impact.png** | `visualize.py` | Experiment 2 dual-axis |
| **plots/exp2_tokens_vs_accuracy.png** | `visualize.py` | Experiment 2 scatter |
| **plots/exp3_rag_vs_full.png** | `visualize.py` | Experiment 3 bars |
| **plots/exp3_rag_improvements.png** | `visualize.py` | Experiment 3 radar |
| **plots/exp4_strategy_comparison.png** | `visualize.py` | Experiment 4 panels |
| **notebook_results.json** | notebook | Results from notebook run |

---

## 🆘 Quick Help

### I get import errors
```bash
pip install numpy pandas matplotlib seaborn
```

### Python command not found
Use `python3` instead of `python`:
```bash
python3 demo.py
```

### Results file not found
Run experiments first:
```bash
python3 context_lab.py
```

### Want to integrate real LLM
See `README.md` section "Integration with Real LLMs"

### Need to understand code
Start with function docstrings in `context_lab.py`

### Writing a report
Use `report_plan.md` as template

---

## 📚 Reading Order by Goal

### Goal: Quick Understanding
1. INDEX.md (this file)
2. QUICK_START.md
3. Run demo.py
4. Done!

### Goal: Full Understanding
1. INDEX.md
2. QUICK_START.md
3. README.md
4. report_plan.md
5. context_lab.py (code)
6. PROJECT_SUMMARY.md

### Goal: Integration
1. QUICK_START.md
2. README.md - Integration section
3. context_lab.py - ollama_query() function
4. Test with your LLM

### Goal: Teaching
1. README.md
2. notebook_template.ipynb
3. Run all experiments
4. Generate plots
5. Use report_plan.md for expected results

### Goal: Research
1. report_plan.md - Methodology
2. context_lab.py - Implementation
3. Run experiments
4. Analyze results
5. Use PROJECT_SUMMARY.md for citations

---

## 💡 Tips

1. **Start Small**: Run `demo.py` first before full experiments
2. **Read Comments**: Code is extensively documented
3. **Check Examples**: README.md has many code examples
4. **Use Notebook**: Great for learning interactively
5. **Customize Gradually**: Start with default parameters
6. **Save Results**: Always save JSON for reproducibility
7. **Generate Plots Early**: Visual results help understanding

---

## 🎯 Key Files by User Type

### Student/Learner
- Start: QUICK_START.md
- Run: demo.py
- Learn: notebook_template.ipynb
- Reference: README.md

### Researcher
- Understand: report_plan.md
- Code: context_lab.py
- Results: PROJECT_SUMMARY.md
- Analysis: visualize.py

### Developer
- Implementation: context_lab.py
- Architecture: PROJECT_SUMMARY.md
- Integration: README.md
- Testing: demo.py

### Teacher/Presenter
- Overview: PROJECT_SUMMARY.md
- Demo: demo.py + notebook_template.ipynb
- Visuals: visualize.py
- Handout: QUICK_START.md

### Manager/Reviewer
- Summary: PROJECT_SUMMARY.md
- Capabilities: README.md
- Results: report_plan.md
- Demo: demo.py output

---

## ✅ Complete File Checklist

- [x] context_lab.py - Implementation
- [x] demo.py - Quick demo
- [x] visualize.py - Plot generation
- [x] notebook_template.ipynb - Interactive
- [x] requirements.txt - Dependencies
- [x] .gitignore - Git rules
- [x] INDEX.md - This navigation guide
- [x] QUICK_START.md - Fast start
- [x] README.md - Full docs
- [x] report_plan.md - Analysis plan
- [x] PROJECT_SUMMARY.md - Overview

**Total: 11 files created** ✅

---

## 🚀 Ready to Start?

**Choose your path**:

- 🏃 **Speed Run** (30 sec): `python3 demo.py`
- 📚 **Learn First** (5 min): Read `QUICK_START.md`
- 🔬 **Full Lab** (15 min): Read `README.md` → Run all
- 💻 **Integrate** (30 min): Read integration docs
- 🎓 **Teach** (1 hr): Run all → Generate plots → Prepare notebook

**Most Common First Step**: `python3 demo.py` then read `QUICK_START.md`

---

**Questions?** Check the documentation files listed above. Each is designed for specific use cases and reading levels.

