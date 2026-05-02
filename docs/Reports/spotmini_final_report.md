# SpotMini Final Report
## The Pennsylvania State University
## College of Engineering
## Senior Capstone Project
## Prepared by Dhruv Nasit and Diviprakash Kuppusamy
## Date: April 2026
<PAGEBREAK>

# ABSTRACT

This report presents the design and evaluation of a SpotMini-inspired quadruped locomotion platform built for senior capstone work. The project goal was to improve robot walking on uneven terrain in simulation by combining a structured Bezier gait baseline with reinforcement learning. Rather than treating the work as a single walking demo, the project grew into a broader experimentation platform that supports simulation setup, training, playback, comparison, manual testing, and dashboard-based inspection.

The final platform uses PyBullet for simulation, Stable-Baselines3 PPO for learning, multiple locomotion environments for flat and rough terrain, and several layers of tooling around model runs. During development, the team added safer playback behavior, four-phase gait support, rough-terrain height residuals, dual-IMU stability variants, teacher-residual transfer logic, and local dashboard launchers. These additions were driven by repeated testing rather than by following one fixed plan from the beginning.

The main result of the project is not a perfect final gait. The more important result is a working robotics experimentation workflow that can train and inspect multiple locomotion strategies while making the causes of failure easier to understand. The capstone shows that rough-terrain locomotion quality depended not only on training longer, but also on improving action authority, reward alignment, gait structure, observation design, and tooling quality at the same time.

<PAGEBREAK>

# TABLE OF CONTENTS

1. Introduction  
2. Literature Survey  
3. Requirement Specifications  
4. System Development  
5. Testing and Evaluation  
6. Conclusion  
7. Future Work  
8. References  
9. Appendices  

<PAGEBREAK>

# LIST OF FIGURES

Figure 1. Operational Flowchart  
Figure 2. Technical Design Flowchart  
Figure 3. System Architecture and Training Pipeline  
Figure 4. PERT Chart  
Figure 5. Gantt Chart  
Figure 6. Critical Path Analysis  
Figure 7. Rough Terrain Training Before and After  
Figure 8. Dashboard Overview  
Figure 9. Manual PyBullet GUI  
Figure 10. Crash Test Safe Failure  
Figure 11. Rough Terrain Walking GIF Still  
Figure 12. Customer / Stakeholder Planning Visual  

# LIST OF TABLES

Table 1. Core Platform Components  
Table 2. Requirement and Constraint Summary  
Table 3. Evaluation Metrics Used for Model Comparison  
Table 4. Risk and Mitigation Summary  

<PAGEBREAK>

# INTRODUCTION

Quadruped robots are most valuable in environments where movement is difficult. Flat-ground walking is not enough if the robot is expected to operate on uneven, cluttered, or unstable surfaces. For this project, the core problem was how to build a SpotMini-style platform that could learn better locomotion in simulation while remaining stable enough to test, compare, and explain.

The project began with a broader application framing around inspection and hazardous-site traversal, but the technical work quickly narrowed toward locomotion. That narrowing was useful rather than limiting. The team found that higher-level mission language meant little if the robot could not stay upright, adapt its gait, and move forward on rough terrain. The final report therefore focuses on locomotion as the enabling capability for later inspection or hazard-response use cases.

The project was motivated by both application value and technical challenge. From an application standpoint, quadrupeds are attractive for inspection and site access because they can potentially go where wheeled systems struggle or where direct human exposure is undesirable (Nasit & Kuppusamy, 2026a; Nasit & Kuppusamy, 2026e). From a technical standpoint, quadruped locomotion is a useful capstone problem because it combines simulation, control, software integration, and machine learning in one system.

A second motivation was accessibility of tooling. PyBullet, Stable-Baselines3, and open-source SpotMini-style repositories reduced the barrier enough that a capstone team could attempt meaningful locomotion work without custom hardware manufacturing in the first semester phase (Coumans & Bai, 2017; Stable-Baselines3 Documentation, n.d.; Rahme, n.d.). That still left many implementation problems, but it made the project feasible.

The main objective was to build a simulation-centered locomotion platform that could train and evaluate improved quadruped walking on rough terrain. This objective broke into five working goals:

- maintain a stable SpotMini-style simulation environment
- use a structured gait generator as the baseline locomotion layer
- train RL policies on top of that baseline for flat and rough terrain
- support training, playback, comparison, and manual debugging in one workflow
- make the results understandable through visual and dashboard tooling

The customer-facing planning material pointed toward use in hazardous or hard-to-access spaces. That meant end users would care about stability, repeatability, and ease of inspection more than about raw novelty. For the capstone itself, the practical stakeholder need became simpler: the platform had to let the team test walking changes quickly, reproduce runs, and explain why one approach behaved better than another.

This report treats the final system as a locomotion experimentation platform rather than as a finished field robot. That framing matches the actual work done. The strongest outcome was not one perfect model, but a chain of improvements across gait generation, reward shaping, rough-terrain sensing, comparison tools, and safer playback behavior. Seen that way, the capstone succeeded by turning a fragile robotics codebase into a more dependable platform for locomotion research at small scale.

<PAGEBREAK>

# LITERATURE SURVEY

Quadruped locomotion is difficult because stability depends on body posture, foot placement, contact timing, and terrain interaction all at once. Classical gait methods remain useful because they provide structure, predictability, and a smaller control problem than fully unconstrained motion learning. For this reason, many practical systems combine a gait generator with some higher-level control or adaptation layer instead of asking learning to invent everything from zero.

Rough terrain increases the difficulty because the robot must react to uneven footholds, body pitch and roll disturbances, and changes in support conditions. A policy that looks acceptable on flat terrain can fail quickly once one foot lands higher or lower than expected. This makes observation design especially important. If the policy cannot sense terrain variation or express leg-specific corrections, it is likely to survive only by moving cautiously or by learning brittle compensations.

Reinforcement learning is attractive here because it can optimize behavior that is hard to hand-tune directly, especially when the robot must trade off forward speed, stability, posture, and terrain adaptation. PPO was selected because it is well documented, widely used, and available through Stable-Baselines3 (Stable-Baselines3 Documentation, n.d.). In this project, RL was not used to replace all locomotion logic. Instead, it was layered over structured gait behavior so that learning refined motion rather than starting from an empty action space.

Simulation-first development made sense for both safety and scope. PyBullet provided fast iteration, low hardware risk, and enough physics realism to expose meaningful failures in posture, slipping, or contact timing (Coumans & Bai, 2017). It also allowed the team to explore many variants that would be hard to test physically in a semester, such as rough-terrain randomization, camera-based sensing, residual swing-height actions, or teacher-residual transfer.

The project borrowed heavily from existing SpotMini-style open-source work, especially the robot structure, inverse-kinematics pattern, and Bezier gait concepts in the `spot_mini_mini` ecosystem (Rahme, n.d.). At the same time, the team expanded the workflow around those foundations by adding modern RL training utilities, local dashboards, run comparison tools, safer playback handling, and multiple rough-terrain environment branches. Published work on gait adaptation and perception-guided locomotion also supported the idea that vision or terrain awareness can improve performance when integrated carefully with control (Dassori et al., 2024; Dassori Walker, 2024).

The most important difference in this capstone was the emphasis on workflow and diagnosability. Many student robotics efforts stop at “a model runs.” This project pushed beyond that by creating a sequence of environment variants, preview artifacts, model comparison utilities, and manual test interfaces that made the learning process more inspectable. The final platform is therefore valuable not only because of the gaits it produced, but because it made it much easier to understand why some training directions failed and why others improved.

<PAGEBREAK>

# REQUIREMENT SPECIFICATIONS

Three stakeholder groups mattered most. The first was the capstone team, which needed a repeatable way to train and inspect locomotion behavior. The second was the academic audience, which needed clear evidence that the system had evolved through real testing rather than only through claims. The third was the application-facing stakeholder represented in the project brief and customer materials, which cared about whether a quadruped platform could plausibly support future inspection or hazardous-site traversal use cases.

The project did not attempt a commercial product study in full detail, but the motivating use case was grounded in real demand for mobile inspection platforms. Boston Dynamics and Unitree both demonstrate that legged robots are already positioned as inspection and industrial-support systems, even if their final deployment contexts differ (Boston Dynamics, n.d.; Unitree Robotics, 2023). That broader relevance justified the capstone focus on stable terrain traversal and visibility into failure modes.

The final platform needed to satisfy several technical requirements:

- the simulator had to initialize reliably and support repeated runs
- the robot needed a usable baseline gait before RL refinement
- training and playback scripts had to load checkpoints and environment settings consistently
- rough-terrain variants needed richer observations than flat-ground walking
- visual debugging had to exist alongside numerical logging

These requirements were interconnected. For example, training could not be evaluated honestly if playback failed due to environment mismatches, and rough-terrain work could not improve much if the policy only had shared global gait parameters with no leg-specific adjustment authority.

The first major constraint was time. A one-semester capstone could not fully solve simulation, training, debugging, UI, and hardware transfer all at once. That forced the team to prioritize locomotion quality and workflow stability over ambitions such as physical deployment.

The second major constraint was software compatibility. The project depended on PyBullet, Python ML packages, simulation code inherited from open-source sources, and multiple local environments. Apple Silicon setup issues made that constraint especially visible, because `pybullet` installation through plain `pip` could fail while Conda-based setup was more dependable. The team therefore had to treat environment management as part of the project work rather than as a one-time setup task.

The third constraint was sample efficiency. Rich rough-terrain sensing, camera rendering, and multiple evaluation passes increased runtime cost significantly. Some promising ideas therefore had to be staged through curriculum-style runs or transfer-based fine-tuning instead of being trained from scratch in the heaviest environment every time.

The project assumed that simulation quality was sufficient to make terrain-aware locomotion development meaningful, that RL could improve performance on top of a structured gait baseline, and that enough software stability could be achieved to compare different models fairly. It also assumed that rough-terrain locomotion was the right near-term proxy for later autonomy goals.

The capstone was considered successful if it produced a system that could do more than one isolated demonstration. A successful outcome meant the platform could train models, replay them, compare them, support manual inspection, and show visible evidence of gait improvement. It also meant the team could explain why a run failed or improved, rather than only reporting a single reward value.

The main risks were unstable gaits, misleading rewards, overfitting to flat terrain, runtime crashes during playback, and the possibility that training time would be consumed by tooling failures instead of locomotion learning. The project addressed these risks through safer run loading, crash-test documentation, smaller smoke tests, visual previews, and multiple environment branches that separated flat-ground experimentation from rough-terrain extensions.

<PAGEBREAK>

# SYSTEM DEVELOPMENT

The original concept was a quadruped platform that could support site inspection in difficult terrain. As development progressed, the concept narrowed into a more realistic capstone scope: build a robust simulation and experimentation pipeline for rough-terrain locomotion. That shift improved the project because it focused effort on the part that actually needed to work first.

System planning evolved from a simple “train a walking model” idea into a layered workflow. The team needed a simulator, a gait baseline, RL environments, training and playback scripts, run storage, comparison utilities, and visual interfaces. Planning documents such as the project brief, requirements notes, and revise-redo submission helped define what counted as useful evidence and what counted as a distraction.

At a high level, the platform works in a loop. The user launches a training or playback script, the script constructs a specific environment variant, the environment uses the gait and control core to generate leg targets, and the simulation executes those targets in PyBullet. RL policies observe the resulting state, receive rewards, and adjust behavior over many rollouts. Visual and numerical artifacts are then used to decide what should change next.

The final platform can be understood as four connected layers:

1. user-facing launchers and debug tools  
2. experiment scripts and RL run management  
3. environment variants and gait/control logic  
4. simulation and robot execution  

This layered structure matters because it explains why the repository became larger than a single training script. Once the project started supporting rough-terrain sensing, multiple gait branches, playback safety, and dashboard-style inspection, the architecture had to be treated as a platform.

![Figure 3. SpotMini system architecture and training pipeline.](/Users/Diviprakash/PSU/spot_mini_mini/docs/Reports/spotmini_system_architecture.png)

PyBullet served as the physics layer for all training and testing. The simulator included a SpotMini-style URDF, terrain variants, and enough control access to support inverse-kinematics-based locomotion and RL overlays. Over time, the environment side became richer: rough-terrain probes, local terrain summaries, leveled camera observations, and dual virtual IMU features were added to give the policy more relevant state information.

One important development was the separation of rough-terrain work into its own modules. `spot_rough_terrain_ml.py` introduced terrain-aware observations and rough-specific rewards, while `rough_terrain_height_residuals` added a more expressive action space with per-leg swing-height residuals. These changes reflected an important lesson from the project: if the policy cannot express independent foot adaptation, longer training alone will not solve rough-terrain traversal.

The locomotion baseline relied on Bezier trajectory generation, inverse kinematics, and tuned gait geometry. This structure reduced the burden on RL by giving the robot a meaningful nominal motion pattern before learning. The team used both manual testers and playback tools to inspect whether gait changes actually helped.

A major branch of this work was the four-phase gait generator, where one leg swings at a time instead of using the earlier trot-style scheduling. That branch was initially unstable and sometimes produced awkward, slow, or backward motion. It improved only after several classical control changes: smaller action ranges, stronger fall penalties, lower target speed, lower body posture, yaw stabilization, and a diagonal crawl sequence. The resulting `four_phase_lowcrawl` run showed that RL performance improved most when gait structure and reward design were corrected together.

Stable-Baselines3 PPO was the main learning algorithm. The training scripts managed seed control, run directories, saved configuration files, checkpoint handling, normalization statistics, and evaluation callbacks. Playback scripts were expanded so that trained models could be tested with better protection against missing normalization stats, mismatched run settings, and platform-specific GUI issues.

The RL pipeline grew through several environment variants. Early flat-ground work focused on stable learning and usable playback. Later work explored rough-terrain modules, height residuals, four-phase crawl training, dual-IMU stability variants, and a teacher-residual approach where a stronger long-walk policy served as a locomotion prior while a new policy learned bounded rough-terrain corrections. This progression showed that the team did not treat RL as a black box; each new branch responded to a specific limitation observed in testing.

Training and playback were treated as part of one continuous workflow. `spot_train_ml.py` created repeatable run folders, `spot_play_ml.py` replayed trained models with saved run settings, and `spot_compare_ml.py` supported side-by-side evaluation. This organization mattered because many apparent “model failures” were really environment or loader mismatches before the tooling was hardened.

Visual evidence became especially important. Preview GIFs, latest preview images, and evaluation plots were added because reward values alone did not tell the team enough about posture, slipping, or gait quality. That decision improved both development and reporting because it created inspectable training history rather than only final models.

The project included two complementary user-facing tools: dashboard-style views for model playback and manual apps for lower-level inspection. Streamlit-based launchers and `pywebview` wrappers made it possible to treat the dashboards more like local applications than like loose development pages. Manual PyBullet and tester tools remained important for debugging because they exposed gait quality directly, independent of RL.

These interfaces mattered for two reasons. First, they improved the team’s ability to inspect models without rewriting scripts each time. Second, they made the platform easier to explain to others. A locomotion project becomes much more convincing when the workflow can be shown clearly instead of only described through terminal logs.

Playback safety became a real development theme. The project encountered normalization mismatches, missing checkpoint artifacts, environment-loading issues, and macOS GUI instability inside native PyBullet rendering. Instead of ignoring those issues, the team added safer fallbacks and clearer run discovery behavior so that model playback would fail more honestly and less destructively.

This safety-oriented cleanup was not separate from the locomotion work. It made locomotion evaluation more trustworthy. If the playback path crashes for infrastructure reasons, it becomes much harder to know whether a model is actually bad. By improving crash behavior, the team improved the validity of later model judgments.

Several design turns mattered most:

- rough-terrain work was split into dedicated modules instead of being forced into the flat-ground environment
- a leveled camera branch was added so the robot saw upcoming terrain in a more stable frame
- four-phase gait support was introduced and then heavily tuned for stability
- rough-terrain policies gained per-leg height residual authority
- teacher-residual transfer was added after direct rough-terrain fine-tuning proved unstable

Taken together, these changes show the actual design philosophy of the project. Progress came from combining better structure, better observations, and better workflow support, not from assuming that PPO alone would discover everything.

<PAGEBREAK>

# TESTING AND EVALUATION

Evaluation used both numerical metrics and visual inspection. Reward, episode length, training speed, and model comparison outputs were useful, but they were not enough on their own. A model could survive longer without moving much, or it could gain reward while still looking unstable. For that reason, playback videos, preview GIFs, and live inspection through the dashboard or PyBullet views were treated as necessary evidence.

The flat-ground and four-phase branches showed this clearly. Some early runs achieved nonzero rewards while still falling quickly or drifting backward. Later runs improved only when reward shaping, target speed, posture, and gait order were corrected alongside training length. The `four_phase_lowcrawl` branch became one of the cleaner examples of success because it reached full 2000-step episodes with a much stronger evaluation score and visibly more coherent crawl behavior.

Rough-terrain behavior improved more slowly than flat-ground behavior because the problem was harder and the observations were more expensive. Several runs demonstrated that surviving the full episode horizon was not the same thing as meaningful traversal. In multiple cases the robot stayed alive while barely moving, over-lifted its front legs, or used awkward compensation patterns that looked numerically acceptable but visually poor.

The most important rough-terrain lesson was that each foot needed more expressive adaptation. Once per-leg swing-height residuals and terrain-conditioned lift targets were introduced, the policy finally had a better way to respond to bumps instead of only using one shared clearance setting. That did not solve the rough-terrain problem completely, but it moved the platform closer to the behavior the team actually wanted.

Crash testing was useful because it separated infrastructure failures from locomotion failures. The project documented cases involving model-environment mismatch, bad comparison candidates, and GUI instability. The main value of these tests was not only that they prevented crashes. They also made the platform easier to trust, because a bad run could fail in a more understandable way instead of producing misleading behavior.

Model comparison was important because the best checkpoint was often not the final checkpoint. Several runs peaked partway through training and then degraded, especially in rough-terrain branches where exploration or policy instability remained high. The comparison workflow therefore emphasized best-model selection, saved evaluation artifacts, and side-by-side testing rather than assuming the last checkpoint was always the best one.

Manual testing played a supporting role throughout the project. `spot_tester.py`, manual leg controls, and PyBullet inspection views helped the team understand whether a gait issue came from bad learning, bad trajectory structure, or simply the wrong posture and step timing. This was especially useful when RL results were ambiguous.

The dashboard and launcher tools were validated mainly through usability rather than through benchmark metrics. Their success criterion was whether they reduced friction in viewing runs, switching models, and inspecting behavior. In practice, the move toward local-window launchers and preview generation made the platform much easier to use during development and much easier to present in a report.

The clearest conclusion from testing is that locomotion quality improved most when the project changed the structure of the learning problem, not just the training duration. Better results came from narrowing the target speed, reshaping rewards, adding per-leg residual control, splitting rough-terrain logic into dedicated modules, and using stronger priors such as teacher-residual transfer. This is a useful capstone result because it shows how robotics ML often depends as much on representation and tooling choices as on the learning algorithm itself.

<PAGEBREAK>

# CONCLUSION AND FUTURE WORK

The SpotMini capstone produced a working locomotion experimentation platform centered on simulation, structured gait generation, and reinforcement learning. The final system does not represent a fully solved rough-terrain walker, but it does represent a meaningful technical progression from fragile initial training scripts toward a platform that can train, replay, compare, and inspect multiple locomotion strategies.

The most important achievement was the combination of several layers that now work together: PyBullet simulation, Bezier-based gait logic, rough-terrain environment variants, PPO training, checkpoint comparison, manual testing, and local dashboard-style tooling. By the end of the project, the team had enough structure to see why some models failed, why some improved, and which changes actually helped.

The project also showed that locomotion quality depends on more than model size or training time. Better behavior came from better action authority, better reward design, better sensing structure, and better platform tooling. That is a solid capstone outcome because it reflects real engineering reasoning rather than only trial-and-error training.

The project changed from an application-framed quadruped idea into a locomotion-centered experimentation platform. That shift made the final system more focused and more honest about what had actually been achieved.

The biggest challenges were rough-terrain instability, tooling mismatches, expensive sensing paths, and platform-specific runtime issues. Each of those problems affected the final design.

The strongest lesson is that robotics ML improves faster when the team treats simulation, control, rewards, observations, and tooling as one system. A good workflow can be just as important as a better policy checkpoint.

<PAGEBREAK>

The most immediate next step is to continue the teacher-residual and terrain-aware residual branches. Those approaches preserve useful locomotion structure while still allowing terrain-specific adaptation.

Future work should expand terrain randomization, evaluate more checkpoint-transfer paths, and stress-test policies on a wider variety of rough surfaces.

The leveled camera branch showed promise, but it was expensive. Future work should explore richer but more efficient terrain representations, including compressed depth summaries or learned terrain encoders.

Additional work should incorporate broader variation in friction, body mass, and terrain shape so the policies become less tied to one simulation configuration.

A longer continuation of the project could investigate sim-to-real transfer, hardware calibration, and whether the current gait and residual structures remain useful on a physical platform.

The dashboard workflow can still be improved through cleaner live metrics, stronger comparison views, and more integrated preview management.

Future documentation should add the remaining planning visuals, dashboard screenshots, and comparative result tables so that the full development history is preserved more clearly.

<PAGEBREAK>

# REFERENCES

1. Boston Dynamics. (n.d.). *Spot*. https://bostondynamics.com/products/spot/  
2. Coumans, E., & Bai, Y. (2017). *PyBullet quickstart guide*. Bullet Physics. https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md  
3. Dassori, I., Adams, M., & Vasquez, J. (2024). *Four-Legged Gait Control via the Fusion of Computer Vision and Reinforcement Learning*. 2024 27th International Conference on Information Fusion (FUSION). https://doi.org/10.23919/FUSION59988.2024.10706406  
4. Dassori Walker, I. A. (2024). *Gait adaptation of quadruped robot via central pattern generator and reinforcement learning* [Undergraduate thesis, Universidad de Chile].  
5. Kuppusamy, D., & Nasit, D. (2026). *README and platform setup notes for the SpotMini capstone repository* [Unpublished course software documentation]. The Pennsylvania State University.  
6. Nasit, D., & Kuppusamy, D. (2026a). *Applications survey and project overview: Quadruped robot locomotion systems* [Unpublished course report]. The Pennsylvania State University.  
7. Nasit, D., & Kuppusamy, D. (2026b). *Quadruped gait prediction project brief* [Unpublished course project brief]. The Pennsylvania State University.  
8. Nasit, D., & Kuppusamy, D. (2026c). *Quadruped gait prediction project specs* [Unpublished course report]. The Pennsylvania State University.  
9. Nasit, D., & Kuppusamy, D. (2026d). *Online search and topic selection* [Unpublished course planning notes]. The Pennsylvania State University.  
10. Nasit, D., & Kuppusamy, D. (2026e). *SpotMini customer / autonomous robotic deployment platform* [Unpublished customer-facing capstone PDF]. The Pennsylvania State University.  
11. Nasit, D., & Kuppusamy, D. (2026f). *Week 12 revise, redo submission* [Unpublished course reflection]. The Pennsylvania State University.  
12. Rahme, M. (n.d.). *Spot Mini Mini OpenAI Gym environment* [GitHub repository and project documentation]. https://github.com/moribots/spot_mini_mini  
13. SpotMini Local Dashboard README. (2026). *SpotMini local dashboard* [Repository documentation]. `/spot_bullet/Sim Display/README.md`  
14. SpotMini Platform Crash-Test Report. (2026). *SpotMini platform crash-test report* [Repository documentation]. `/spot_bullet/CRASH_TEST_REPORT.md`  
15. Stable-Baselines3 Documentation. (n.d.). *PPO*. https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html  
16. Unitree Robotics. (2023). *Unitree quadruped robotics platform*. https://www.unitree.com/  

<PAGEBREAK>

# APPENDICES

## Appendix A. Figure and Media Checklist

- system architecture figure included in this report
- flowcharts, dashboard screenshots, and GUI stills can be inserted if required
- rough-terrain GIF stills and comparison images remain optional supporting visuals

## Appendix B. Repository Assets Referenced

- `docs/flowcharts/`
- `docs/Reports/`
- `spot_bullet/media/`
- `spot_bullet/crash_test_screenshots/`
