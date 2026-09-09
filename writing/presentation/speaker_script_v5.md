# S.P.A.S. Speaker Script, version 5

Spoken text for `SPAS_presentation_v5.pptx`, one section per slide. The same text is stored in the PowerPoint notes pane of each slide, so it is visible in presenter view.

Slides 6 to 11 of version 5 carry short bullet fragments, not sentences. The sentences are here instead: the slide shows the claim, the speaker supplies the reasoning. Do not read the bullets aloud, the audience has already read them.

Target length: about 14 minutes of speaking, plus questions. Timings are a guide, not a script to race against.

Regenerate this file with `py speaker_notes_v5.py`. Edit `speaker_notes_v5.py`, not this file.

## Running order

| Part | Slides | Content | Time |
| --- | --- | --- | --- |
| Opening | 1 to 2 | Team, task, outline. | 1:50 |
| Problem and platform | 3 to 4 | Objective, sensor constraints, hardware and software. | 2:35 |
| Control and manipulation | 5 to 6 | State machine, the error term, the grasp sequence. | 3:20 |
| Perception and demonstration | 7 to 8 | The reference run and the two detectors. | 2:25 |
| Results and closing | 9 to 11 | Measurements, limitations, conclusion, questions. | 4:20 |
| **Total** | 1 to 11 | | **14:30** |

## Script

### Slide 1, Self-Propelled Arm System

*About 0:40.*

> **Cue.** Title slide up before you start speaking. Say the team name, not the acronym letters.

Good morning, and thank you for coming. We are the S.P.A.S. team, and our project is the Self-Propelled Arm System.

It is a small tracked robot with a robotic arm on top. Its job is to find drink cans on the floor of an indoor room, pick them up, and drop them into a collection bin.

The interesting part of the project is the constraint we worked under. The robot has one camera and no other sensor. Everything you will see today is decided from that single image.

I will start with a short overview, and then my colleagues will take you through the system and the results.

### Slide 2, Introduction and Outline

*About 1:10.*

> **Cue.** Point at the photograph on the left, then run down the outline on the right.

Here is the system in three lines.

First, it is an autonomous can collection robot: a tracked indoor platform with a four link arm and a parallel gripper.

Second, and this is the important one, a single RGB camera is the only sensor. There is no depth sensor, no range finder, no wheel encoders, and no force sensing in the gripper.

Third, because of that, vision is in the loop at every stage. Search, approach, grasp and delivery are all decided from the image.

The photograph on the left shows the environment we work in: drink cans standing on the floor, and a bin marked with an AprilTag.

The outline on the right is how we will proceed. We start with the problem and the platform, then the architecture and the control logic, then perception and a complete run on hardware, then the measured results, and we close with the limitations and the conclusion.

### Slide 3, Problem Statement

*About 1:20.*

> **Cue.** Left column is the four stages, right column is the sensor list. Land on the bottom right line.

First, why this task.

Drink cans accumulate in indoor public spaces, and collecting them is repetitive, low risk manual work. For us it was a good project task because it combines everything at once: perception, navigation, decision making and manipulation.

The mission breaks into four stages, shown on the left.

In detection, the robot rotates in place until the detector accepts a target. In approach, it steers on the image error and stops when the target looks large enough. In grasp, it lowers the arm, pushes, closes and lifts. In delivery, it docks on the AprilTag and releases the can above the bin.

On the right is the hardware we had to do this with. One RGB camera, 320 by 240 pixels, with a 60 degree field of view. No depth sensing, no stereo pair, no range finder. No wheel encoders, so the base returns no motion feedback at all. And no force sensing, so the gripper cannot detect contact.

The consequence is at the bottom right, and it is the sentence that shapes the whole system: the base is an open loop actuator. We command it to move, and the only way to confirm that it moved is to look at the next camera image.

### Slide 4, System Architecture

*About 1:15.*

> **Cue.** Trace the diagram outward from the Jetson in the middle, then drop to the software stack.

This is the hardware architecture.

In the middle is the Jetson Nano with four gigabytes of shared memory. It runs everything on board. There is no external computer and no remote control.

Three subsystems are connected to it. On the left, the CSI camera delivers 320 by 240 frames over CSI-2. On the right, two track motors are driven with PWM through a motor driver, and they report nothing back. Below them, five bus servos are connected in one TTL serial chain, and these are the only devices in the system whose position we can read. Servo four closes the gripper.

The dashed lines are the power rail, from the battery to the drive and to the computer.

At the bottom is the software stack, in four layers. The mission layer is our demo_core state machine with sixteen states. The perception layer uses detectNet, the pupil_apriltags library and OpenCV. The hardware layer wraps the jetbot Robot and Camera classes and the TTLServo driver. And all tuning constants live in two JSON configuration files, so no threshold is hard coded in the logic.

### Slide 5, State Machine and Control Logic

*About 1:45.*

> **Cue.** Give the graph two seconds of silence before speaking. Read the formula aloud slowly.

The mission logic is an explicit finite state machine, sixteen states in total. The graph on the left is the design level view. The guards on the transitions are fields of a single shared context object, so every state reads the same picture of the world.

On the right are the four states that are driven by vision.

In searching, the robot rotates until a target is accepted or the timeout expires. In aligning, it turns until the target is within tolerance, and it aborts if the target is lost. In approaching, it steers or drives forward and stops on apparent size. Final verification repeats the check with a tighter tolerance over several consecutive frames, so a single lucky frame cannot trigger a grasp.

The formula below is the whole control input. e_x is the horizontal position of the target centre, minus half the image width, divided by the image width. It ranges from minus zero point five to plus zero point five, negative to the left of centre. We steer to drive that error to zero.

Distance is handled the same way. We do not estimate range. We stop when the detection box reaches 0.36 of the image height for a can, or 0.15 for the tag. Apparent size replaces a distance measurement.

Two design decisions at the bottom. Transitions are held in an explicit table, so an undeclared state and event pair raises an error instead of failing silently. And finalising requires a stable target, which map based navigation never provides, so dead reckoning alone can never trigger a grasp.

### Slide 6, Grasping Strategy

*About 1:35.*

> **Cue.** Walk the six chips left to right, then the four photographs, then the two blocks on the right.

Now the grasp itself.

The line at the top is the whole idea: fixed poses, no inverse kinematics. The arm only ever moves between poses we defined in advance, and the base drives the can into the open gripper.

The six chips are the sequence. Two seconds of settling. Lower the arm to the down pose, polling the servos as it goes. Push forward slowly, which is the step highlighted in blue. Hold the arm locked for five seconds. Close to the grab pose. Lift to the carry pose.

The four frames below are recorded on board during a real pickup: verify, push, close, carry.

Then the question on the right: why push, rather than reach out and grasp? The arm is short and the gripper closes imprecisely, but the base moves reliably over short distances. So we lower the open gripper to the floor and drive the can into it. We trade fine manipulation, which our hardware is bad at, for coarse base motion, which it does well.

The block underneath is the one place in the whole system with real feedback. Servo positions are polled until the readings stop changing, instead of waiting out a fixed delay, and the push is blocked while the arm has not settled. In the archived run that came to 4.94 seconds of settling, 7 seconds of pushing and 5 seconds of holding.

The limitation is on the slide and we will not talk around it: this works for light objects on a low friction floor.

### Slide 7, Experimental Setup

*About 1:05.*

> **Cue.** Three photographs first, then the four numbers. This is the slide to slow down on.

This is the experimental setup and one complete reference run.

On the left, the platform holding a grasped can. In the middle, the arena: a 250 millilitre can and the bin with the AprilTag. On the right, the end of the run, with the can held above the bin.

The four numbers describe that run. 73.8 seconds from the start of the search to the release over the bin. One camera, and nothing else: no depth sensor, no range finder, no encoders. 98 can detections during the run. And 41 successful tag reads out of 59 attempts.

So search, grasp, transport and release were all completed on real hardware using the camera alone. That is not a claim we are asking you to take on trust: the on-board overlay records DEPTH equals NO on every single frame of that run.

### Slide 8, Object and Marker Detection

*About 1:20.*

> **Cue.** Point at the overlay values in the middle image; those are the two numbers the controller reads.

Perception is two detectors, and between them they produce every measurement the controller uses.

On the left is the object detector, SSD-MobileNet-V1, at 300 by 300 input resolution, with the confidence score.

In the middle is the same detector running inside the mission, with our overlay on top. The two values under the image are exactly what the controller consumes: a horizontal error of minus 0.119, so the can is slightly left of centre, and a box height of 0.255, which is still below the 0.36 stopping threshold. So at this moment the robot keeps approaching.

On the right is the AprilTag detector during docking at the bin. Family tag36h11, identifier zero, and a tag height of 0.186 against the 0.15 threshold.

The heading at the bottom says no calibration needed, and that is worth one sentence. We never calibrated this camera. Tag height, edge ratio and corner angle are all ratios measured inside the image, so the focal length cancels out of every one of them. The AprilTag library will also hand us a metric pose through perspective-n-point, and we deliberately do not use it.

### Slide 9, Experimental Results

*About 1:35.*

> **Cue.** Left chart, then the four numbers, then the caveat box on the right. Do not skip the caveat.

Now the measurements.

The left chart is detector accuracy as F1 score. The two bars in each group compare the model we actually deployed against the stronger custom model. On photographs we give up about six F1 points, and on video about two.

We accepted that, and the three bullets underneath are the reason. The deployed model drops a custom post-processing layer and a custom inference backend, and it runs on the inference path NVIDIA supports natively on the Nano. We traded a small amount of accuracy for a deployment that works reliably on the device we have. The conversion itself is lossless in practice: the deviation between the PyTorch model and the exported ONNX model is 4.4 times ten to the minus seven.

The right chart is where the time goes in the 73.8 second run. Only 31.2 seconds, about 42 percent, is commanded base motion. Another 11.9 seconds are deliberate holds, the settling and locking we saw in the grasp sequence. The remainder is perception and decision time.

One caveat, and it is on the slide rather than hidden in a footnote. That run was recorded by the earlier prototype, with the legacy parameter set and the custom TensorRT detector. Its stop thresholds were 0.400 and 0.500, not the 0.36 and 0.15 we use today. The counts and the timings are exact; the thresholds behind them are not the current ones.

### Slide 10, Limitations and Challenges

*About 1:35.*

> **Cue.** Three orange cards left to right. Pause on the third one, it is the honest one.

We want to be clear about what does not work.

First card, depth excluded. We tested a depth network. It was reliable to about 0.3 metres in a general scene, and to about one metre only on clean white surfaces. Our can detector already works out to three metres, so the depth estimate was worse than the detector it was meant to help. We took it out of the control path completely.

Second card, the pose estimate drifts. The two tracks turn at different rates, somewhere between one and three radians per second, while our model assumes fixed values of 0.471 and 0.942. The error is larger than the quantity being modelled, so dead reckoning is not trustworthy, and the AprilTag on the bin is the only absolute anchor the robot has.

Third card, and this is the one that worries us, a silent failure. Grasp verification is currently disabled, so a pickup is recorded whether or not the can is in the gripper. The robot can report success holding nothing. Enabling that verification flag is the first item on our list.

Taken together, the missing capability is relocalisation. The map can direct the robot's attention, but it can never trigger an action on its own.

The row at the bottom is the practical side. Open loop turning was unreliable, so we replaced it with short pulses and visual correction. The gripper gives no feedback, so full and empty are indistinguishable. We get 10 to 15 frames per second, so motion is pulsed and speeds stay low. Deployment cost accuracy and bought reliability. Depth underperformed. And grasp timing needed real feedback, because a fixed delay simply did not work.

### Slide 11, Conclusion and Future Work

*About 1:10.*

> **Cue.** Summary on the left, sensors on the right. Stop talking on the thank you line.

To summarise.

We built an indoor mobile manipulator that runs on a single camera. Its control loops close on how the target appears in the image, not on estimated geometry. The practical effect is the third line: errors degrade performance instead of breaking the mission. When the map drifts, the robot searches longer. When turning is asymmetric, it makes a few more corrections. It still finishes.

The one exception is in the blue box, the undetected empty gripper, and we have said why. In general, on hardware with no proprioception, we would rather be conservative and slow than confidently wrong.

The next steps are deliberately small. Enable grasp verification, which costs one extra observation per pickup. And calibrate the two turn constants, because the per direction scale factors already exist in the configuration and are simply left at one.

If we had a budget for sensors, the order is on the right. A forward distance sensor, which would be the first true metric measurement in the system. Force or current sensing in the gripper, which addresses the empty gripper failure directly. Wheel odometry, which turns the map from a hint into a measurement. And more tags around the room as cheap relocalisation anchors.

Thank you for your attention. We are happy to take your questions.

## If you are running long

Cut in this order. The right column is what still has to be said.

| Cut | Keep |
| --- | --- |
| Slide 4, software stack | Name the four layers in one sentence and move on. |
| Slide 5, the two design decisions at the bottom | The formula and the two thresholds have to stay. |
| Slide 9, the ONNX deviation figure | Keep the F1 trade and the 42 percent. |
| Slide 10, the six challenge rows | The three cards have to stay. |
| Slide 11, the sensor list | Name the distance sensor only. |

## Questions to expect

Short answers, grounded in what the deck shows. Where the project has no measurement, say so rather than estimating.

**Why not just add a depth sensor or use the depth network?**

We tested the depth network: about 0.3 metres in a general scene, about one metre on clean white surfaces. The can detector reaches three metres, so depth was worse than what it was meant to support. A hardware distance sensor is the first item on our sensor list for exactly that reason.

**How does the robot know how far away the can is, with no range sensor?**

It does not, and it does not need to. It stops on apparent size: the detection box reaching 0.36 of the image height for a can, 0.15 for the tag. Apparent size stands in for distance.

**You never calibrated the camera. Does that not bias the measurements?**

Every quantity we use is a ratio measured inside the image: tag height, edge ratio, corner angle. The focal length cancels in all of them. The AprilTag library can give a metric pose through perspective-n-point, and we deliberately do not use it.

**Why push the can into the gripper instead of grasping it?**

The arm is short and the gripper closes imprecisely; the base drives reliably over short distances. We trade the motion our hardware is bad at for the motion it is good at. The cost is on the slide: light objects, low friction floor.

**What happens if the gripper closes on nothing?**

Today the run is recorded as a successful pickup, because grasp verification is disabled. That is the silent failure on slide 10 and the first fix on our list. It costs one extra observation per pickup.

**Why deploy the weaker detector?**

Six F1 points on photographs and two on video, in exchange for dropping a custom post-processing layer and a custom backend and running on the path NVIDIA supports natively on the Nano. The export itself is lossless: 4.4 times ten to the minus seven between PyTorch and ONNX.

**41 tag reads out of 59 attempts is a two thirds hit rate. Is that a problem?**

In that run, no: a failed read costs one frame and the next frame is tried, and the docking completed. We did not log a cause for each individual failure, so we cannot break those 18 misses down for you.

**Why is only 42 percent of the run spent moving?**

31.2 seconds is commanded base motion, 11.9 seconds are deliberate holds from the grasp sequence, and the rest is perception and decision time at 10 to 15 frames per second.

**Does the map do anything?**

It directs attention, and it can never trigger an action. Finalising a grasp requires a stable visual target, which dead reckoning cannot provide. That is deliberate, given the drift on slide 10.

**How well does it work in different lighting, or with other objects?**

Not measured. We evaluated the detector on our photograph and video sets and ran the full mission on the arena you saw. Lighting robustness and other object classes are outside what we tested.

**What is the success rate over many runs?**

We report one instrumented end to end run, 73.8 seconds, with the counts and timings on slide 9. We do not have a trial count large enough to quote a success rate, and with grasp verification disabled we could not score those trials reliably anyway.

**What would you do differently?**

Enable grasp verification from the start, and calibrate the two turn constants rather than leaving both scale factors at one. Both are small changes; both were left too late.
