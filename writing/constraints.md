## Some observed constraints
1. The tank turn **asymmetrically** to the left and right with same angle parameters with an difference of velocities **1 rad< diff < 3 rad/s**.
2. "Sensor range" for the neural models, for depth net is about **0.3 meter**, for ssd-moblinet is **3 meters**. 
3. When the tank move slowly with velocity v(can not directly read though) or power p, if the friction coefficient μ w.r.t. the ground and the redbull can is less than some constant k, then **the can will not fall during picking**.
4. If the background is white and clean, the depth net can work in a range of **1 meter**
5. The tank has w cm width.
6. We can control the speed of the claw.
7. The claw can only stably grab the redbull up **iff the redbull can is completely hold in := the distance to the redbull's center is smaller than k cm**(not just holding but also close to the tank side).
8. The claw can reach 20 cm heightest, the redbull can has about 10cm height, the bin has about 5 cm height.
9. The computer can render stably 10~15 FPS.

## Something worth to show(invariants)
1. tba
