import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
from panda3d.core import Vec3
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import ZUp
from panda3d.core import *
from panda3d.physics import *
from direct.actor.Actor import Actor
from panda3d.bullet import BulletDebugNode
import Drone

class DroneEnv():

    def __init__(self):

        self.visualize = False
        self.actors = 2

        #repair this in drone class as well
        if self.visualize == False :
            from pandac.PandaModules import loadPrcFileData
            loadPrcFileData("", "window-type none")

        import direct.directbase.DirectStart

        self.ep = 0
        self.ep_rew = 0

        self.t = 0

        self.action_space = Box(-1,1,shape=((self.actors*3),))
        self.observation_space = Box(-50,50,shape=((self.actors*3 + 3),))

        self.target = 8*np.random.rand(3)
        self.construct()

        self.percentages = []
        self.percentMean = []
        self.percentStd = []

        taskMgr.add(self.stepTask, 'update')
        taskMgr.add(self.lightTask, 'lights')

        self.forces = np.array([0,0,9.81]*self.actors, dtype = np.float)

    def construct(self) :

        if self.visualize :

            base.cam.setPos(17,17,1)
            base.cam.lookAt(0, 0, 0)

            wp = WindowProperties()
            wp.setSize(1200, 500)
            base.win.requestProperties(wp)

        # World
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

        #skybox
        skybox = loader.loadModel('../models/skybox.gltf')
        skybox.setTexGen(TextureStage.getDefault(), TexGenAttrib.MWorldPosition)
        skybox.setTexProjector(TextureStage.getDefault(), render, skybox)
        skybox.setTexScale(TextureStage.getDefault(), 1)
        skybox.setScale(100)
        skybox.setHpr(0, -90, 0)

        tex = loader.loadCubeMap('../textures/s_#.jpg')
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

        #multiple drone initialization

        wrgDistance = True
        positions = []

        self.uavs = [Drone.uav() for i in range(self.actors)]
        [self.world.attachRigidBody(uav.drone) for uav in self.uavs]
        [uav.drone.setDeactivationEnabled(False) for uav in self.uavs]

        #target object
        self.targetObj = loader.loadModel("../models/target.gltf")
        self.targetObj.reparentTo(render)
        self.targetObj.setPos(Vec3(self.target[0], self.target[1], self.target[2]))

    def lightTask(self, task) :

        count = globalClock.getFrameCount()

        for uav in self.uavs :

            rest = count % 100
            if rest < 10 :
                uav.plight2.setColor((0.1, 0.9, 0.1, 1))
            elif rest > 30 and rest < 40 :
                uav.plight2.setColor((0.9, 0.1, 0.1, 1))
            elif rest > 65 and rest < 70 :
                uav.plight2.setColor((0.9,0.9, 0.9, 1))
            elif rest > 75 and rest < 80 :
                uav.plight2.setColor((0.9,0.9, 0.9, 1))
            else :
                uav.plight2.setColor((0.1, 0.1, 0.1, 1))

        return task.cont


    def getState(self) :

        stateVec = []

        for uav in self.uavs :
            pos = uav.drone.transform.pos
            stateVec.append(pos)

        stateVec.append(self.target)
        state = np.array(stateVec).reshape((3*self.actors + 3),)
        state = np.around(state, decimals = 2)

        state = np.expand_dims(state, 0)

        return state

    def getReward(self) :

        reward1 = 0
        reward2 = 0
        distance = 0

        pos = self.uavs[0].drone.transform.pos
        d = np.linalg.norm(pos - self.target)

        if d < 5 :
            reward1 = 5 - d
            reward1 = reward1/20

        pos = self.uavs[1].drone.transform.pos
        d = np.linalg.norm(pos - self.target)

        if d < 5 :
            reward2 = 5 - d
            reward2 = reward2/20

        return reward1, reward2


    def reset(self):

        #log
        self.percentages.append(self.ep_rew)
        me = np.mean(self.percentages[-500:])
        self.percentMean.append(me)
        self.percentStd.append(np.std(self.percentages[-500:]))


        print(f'{self.ep}   {self.t}    {self.ep_rew:+8.2f}    {me:+6.2f}')

        prevPos = 8*(np.random.rand(3)-0.5) + 8*(np.random.rand(3)-0.5) - np.array([0,0,2], np.float32)
        #physics reset
        for uav in self.uavs :
            inPos = prevPos + np.array([0,0,4], np.float32)
            #print(f'{inPos}     {prevPos}')
            prevPos = np.copy(inPos)

            uav.body.setPos(inPos[0], inPos[1], inPos[2])
            uav.body.setHpr(0, 0, 0)
            uav.drone.set_linear_velocity(Vec3(0,0,0))
            uav.drone.setAngularVelocity(Vec3(0,0,0))

        self.forces = np.array([0,0,9.81]*self.actors, dtype = np.float)

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

        if self.visualize :
            self.world.doPhysics(dt)
        else :
            self.world.doPhysics(0.1)

        for i, uav in enumerate(self.uavs) :
            force = self.forces[i*3: (i+1)*3]
            forceVec3 = Vec3(force[0], force[1], force[2])
            uav.drone.applyCentralForce(forceVec3)

        return task.cont


    def step(self, action):

        done = False
        reward = 0

        self.t += 1
        reward1, reward2 = self.getReward()
        self.ep_rew += reward1 + reward2
        state = self.getState()

        self.forces += action
        basis = np.array([0,0,9.81]*self.actors, dtype = np.float)

        #10 sub steps in each step
        for i in range(10) :
            c = taskMgr.step()
            self.forces -= 0.05*(self.forces - basis)

            #collision test
            if self.world.contactTestPair(self.uavs[0].drone, self.uavs[1].drone).getNumContacts() > 0 :
                done = True
                break

        #time constraint
        if self.t > 200 :
            done = True

        #position constraint :
        for uav in self.uavs :
            pos = uav.drone.transform.pos
            if np.max(np.abs(pos)) > 49 :
                done = True

        return state, reward1, reward2, done, {}
