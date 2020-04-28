import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt

import direct.directbase.DirectStart
from panda3d.core import Vec3
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape
from panda3d.core import *
from panda3d.physics import *
from direct.actor.Actor import Actor
from panda3d.bullet import BulletDebugNode

class DroneEnv(gym.Env):

    def __init__(self,config):

        self.ep = 0
        self.ep_rew = 0
        self.t = 0

        self.action_space = Box(-1,1,shape=(3,))
        self.observation_space = Box(-50,50,shape=(9,))


        self.target = 8*np.random.rand(3)
        self.construct()

        self.percentages = []
        self.percentMean = []
        self.percentStd = []

        taskMgr.add(self.stepTask, 'update')
        taskMgr.add(self.lightTask, 'lights')

        self.rotorForce = np.array([0,0,9.81], dtype = np.float)

    def construct(self) :

        base.cam.setPos(17,17,1)
        base.cam.lookAt(0, 0, 0)

        wp = WindowProperties()
        wp.setSize(1200, 500)
        base.win.requestProperties(wp)

        # World
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

        #skybox
        skybox = loader.loadModel('models/skybox.gltf')
        skybox.setTexGen(TextureStage.getDefault(), TexGenAttrib.MWorldPosition)
        skybox.setTexProjector(TextureStage.getDefault(), render, skybox)
        skybox.setTexScale(TextureStage.getDefault(), 1)
        skybox.setScale(100)
        skybox.setHpr(0, -90, 0)

        tex = loader.loadCubeMap('textures/s_#.jpg')
        skybox.setTexture(tex)
        skybox.reparentTo(render)

        render.setTwoSided(True)

        #Light
        plight = PointLight('plight')
        plight.setColor((1.0, 1.0, 1.0, 1))
        plnp = render.attachNewNode(plight)
        plnp.setPos(0, 0, 0)
        render.setLight(plnp)

        # Create Ambient Light
        ambientLight = AmbientLight('ambientLight')
        ambientLight.setColor((0.15, 0.05, 0.05, 1))
        ambientLightNP = render.attachNewNode(ambientLight)
        render.setLight(ambientLightNP)


        # Drone
        shape = BulletBoxShape(Vec3(0.5, 0.5, 0.5))
        self.drone = BulletRigidBodyNode('Box')
        self.drone.setMass(1.0)
        self.drone.addShape(shape)
        self.droneN = render.attachNewNode(self.drone)
        self.droneN.setPos(0, 0, 3)
        self.world.attachRigidBody(self.drone)
        model = loader.loadModel('models/drone.gltf')
        model.setHpr(0, 90, 0)
        model.flattenLight()
        model.reparentTo(self.droneN)

        blade = loader.loadModel("models/blade.gltf")
        blade.reparentTo(self.droneN)
        blade.setHpr(0, 90, 0)
        blade.setPos(Vec3(0.3, -3.0, 1.1))
        rotation_interval = blade.hprInterval(0.2,Vec3(180,90,0))
        rotation_interval.loop()

        placeholder = self.droneN.attachNewNode("blade-placeholder")
        placeholder.setPos(Vec3(0, 6.1, 0))
        blade.instanceTo(placeholder)

        placeholder = self.droneN.attachNewNode("blade-placeholder")
        placeholder.setPos(Vec3(3.05, 3.0, 0))
        blade.instanceTo(placeholder)

        placeholder = self.droneN.attachNewNode("blade-placeholder")
        placeholder.setPos(Vec3(-3.05, 3.0, 0))
        blade.instanceTo(placeholder)

        #drone ligths
        self.plight2 = PointLight('plight')
        self.plight2.setColor((0.9, 0.1, 0.1, 1))
        plnp = self.droneN.attachNewNode(self.plight2)
        plnp.setPos(0, 0, -1)
        self.droneN.setLight(plnp)

        #over light
        plight3 = PointLight('plight')
        plight3.setColor((0.9, 0.8, 0.8, 1))
        plnp = self.droneN.attachNewNode(plight3)
        plnp.setPos(0, 0, 2)
        self.droneN.setLight(plnp)

        #target object
        self.targetObj = loader.loadModel("models/target.gltf")
        self.targetObj.reparentTo(render)
        #self.targetObj.setPos(Vec3(1, 1, 2))
        self.targetObj.setPos(Vec3(self.target[0], self.target[1], self.target[2]))

    def lightTask(self, task) :

        count = globalClock.getFrameCount()

        rest = count % 100
        if rest < 10 :
            self.plight2.setColor((0.1, 0.9, 0.1, 1))
        elif rest > 30 and rest < 40 :
            self.plight2.setColor((0.9, 0.1, 0.1, 1))
        elif rest > 65 and rest < 70 :
            self.plight2.setColor((0.9,0.9, 0.9, 1))
        elif rest > 75 and rest < 80 :
            self.plight2.setColor((0.9,0.9, 0.9, 1))
        else :
            self.plight2.setColor((0.1, 0.1, 0.1, 1))

        return task.cont

    def getState(self) :

        vel = self.drone.get_linear_velocity()
        po = self.drone.transform.pos
        ang = self.droneN.getHpr()

        velocity = np.nan_to_num(np.array([vel[0], vel[1], vel[2]]))
        position = np.array([po[0], po[1], po[2]])

        state = np.array([position, self.target, velocity]).reshape(9,)
        state = np.around(state, decimals = 2)  #experimental, decimal places

        return state

    def getReward(self) :

        reward = 0

        s = self.getState()

        d = np.linalg.norm(s[0:3] - s[3:6])

        if d < 5 :
            reward = 5 - d
            reward = reward/20

        if np.sum(s[6:9]) == 0 :
            reward = -0.1

        return reward

    def reset(self):

        #log
        self.percentages.append(self.ep_rew)
        me = np.mean(self.percentages[-500:])
        self.percentMean.append(me)
        self.percentStd.append(np.std(self.percentages[-500:]))

        s = self.getState()
        d = np.linalg.norm(s[:3] - self.target)
        ds = np.linalg.norm(s[:3] - np.array([0,0,4]))


        if self.ep %50 == 0 :
            self.PlotReward()

        dv = np.mean(np.abs(s[3:6]))

        print(f'{self.ep}   {self.t}    {self.ep_rew:+8.2f}    {me:+6.2f}    {d:6.2f}    {ds:6.2f}    {dv:6.2f}') #{s[:6]}

        #physics reset
        self.droneN.setPos(0,0,4)
        self.droneN.setHpr(0, 0, 0)
        self.drone.set_linear_velocity(Vec3(0,0,0))
        self.drone.setAngularVelocity(Vec3(0,0,0))

        self.rotorForce = np.array([0,0,9.81], dtype = np.float)

        #define new target:
        self.target = 8*(2*np.random.rand(3)-1)
        self.target[2] = np.abs(self.target[2])
        self.targetObj.setPos(Vec3(self.target[0], self.target[1], self.target[2]))


        self.ep +=1
        self.t = 0
        self.ep_rew = 0

        state = self.getState()

        return state

    def stepTask(self, task) :

        dt = globalClock.getDt()
        self.world.doPhysics(dt)

        #application of force
        force = Vec3(self.rotorForce[0], self.rotorForce[1], self.rotorForce[2])
        self.drone.applyCentralForce(force) #should be action

        po = self.drone.transform.pos
        position = np.array([po[0], po[1], po[2]])

        #self.smoke.setPos(Vec3(position[0], position[1], position[2]))

        return task.cont


    def step(self, action):

        done = False
        reward = 0

        self.t += 1
        reward = self.getReward()
        self.ep_rew += reward
        state = self.getState()

        self.rotorForce += action #0.2*action

        basis = np.array([0,0,9.81], dtype = np.float)

        #10 sub steps in each step
        for i in range(10) :
            c = taskMgr.step()
            self.rotorForce -= 0.05*(self.rotorForce -basis)

        #time constraint
        if self.t > 50 :
            done = True

        return state, reward, done, {}

    def PlotReward(self) :

        c = range(len(self.percentages))
        plt.plot(self.percentMean, c= 'b', alpha = 0.8)
        plt.fill_between(c, np.array(self.percentMean)+np.array(self.percentStd), np.array(self.percentMean)-np.array(self.percentStd), color='g', alpha=0.3, label='Objective 1')

        plt.grid()
        plt.savefig('rews.png')
        plt.close()
