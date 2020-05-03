import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
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

        self.visualize = False

        if self.visualize == False :
            from pandac.PandaModules import loadPrcFileData
            loadPrcFileData("", "window-type none")

        import direct.directbase.DirectStart

        self.ep = 0
        self.ep_rew = 0
        self.t = 0

        self.action_space = Box(-1,1,shape=(4,))
        self.observation_space = Box(-50,50,shape=(6,))

        self.target = 8*np.random.rand(3)
        self.construct()

        self.percentages = []
        self.percentMean = []
        self.percentStd = []

        taskMgr.add(self.stepTask, 'update')
        taskMgr.add(self.lightTask, 'lights')

        self.forceDiff = np.array([1,1,1,1], dtype = np.float)/4

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

        # Drone
        shape = BulletBoxShape(Vec3(0.5, 0.5, 0.5))
        self.drone = BulletRigidBodyNode('Box')
        self.drone.setMass(1.0)
        self.drone.addShape(shape)
        self.droneN = render.attachNewNode(self.drone)
        self.droneN.setPos(0, 0, 3)
        self.world.attachRigidBody(self.drone)
        model = loader.loadModel('../models/drone.gltf')
        model.setHpr(0, 90, 0)
        model.flattenLight()
        model.reparentTo(self.droneN)

        blade = loader.loadModel("../models/blade.gltf")
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
        self.targetObj = loader.loadModel("../models/target.gltf")
        self.targetObj.reparentTo(render)
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

        #vel = self.drone.get_linear_velocity()
        po = self.drone.transform.pos
        #ang = self.droneN.getHpr()

        ang = render.getRelativeVector(self.droneN,Vec3(0,0,1))
        normal = np.array([ang[0], ang[1], ang[2]])

        #velocity = np.nan_to_num(np.array([vel[0], vel[1], vel[2]]))
        position = np.array([po[0], po[1], po[2]])

        state = np.array([position, normal]).reshape(6,)
        state = np.around(state, decimals = 3)

        return state

    def getReward(self) :

        reward = 0.01

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

        self.forceDiff = np.array([0,0,0,0], dtype = np.float) #np.array([1,1,1,1], dtype = np.float)/4

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

        if self.visualize :
            self.world.doPhysics(dt)
        else :
            self.world.doPhysics(0.1)

        self.drone.setDeactivationEnabled(False)

        #inclination
        ang = render.getRelativeVector(self.droneN,Vec3(0,0,1))
        normal = np.array([ang[0], ang[1], ang[2]])

        if np.sum(normal) > 0 :
            normal = normal/np.sum(normal)

        self.drone.applyImpulse(Vec3(self.forceDiff[0]*normal[0], self.forceDiff[0]*normal[1], self.forceDiff[0]*normal[2]), Point3(3,0,1))
        self.drone.applyImpulse(Vec3(self.forceDiff[1]*normal[0], self.forceDiff[1]*normal[1], self.forceDiff[1]*normal[2]), Point3(-3,0,1))
        self.drone.applyImpulse(Vec3(self.forceDiff[2]*normal[0], self.forceDiff[2]*normal[1], self.forceDiff[2]*normal[2]), Point3(0,3,1))
        self.drone.applyImpulse(Vec3(self.forceDiff[3]*normal[0], self.forceDiff[3]*normal[1], self.forceDiff[3]*normal[2]), Point3(0,-3,1))


        return task.cont

    def step(self, action):

        done = False
        reward = 0

        self.t += 1
        reward = self.getReward()
        self.ep_rew += reward
        state = self.getState()

        basis = np.array([1,1,1,1], np.float32)/250
        self.forceDiff = basis + 0.01*action
        taskMgr.step()

        #for i in range(10) :
        #    self.forceDiff = basis
        #    taskMgr.step()

        #time constraint
        if self.t > 500 :
            done = True

        #inclination constraint
        ang = render.getRelativeVector(self.droneN,Vec3(0,0,1))
        normal = np.array([ang[0], ang[1], ang[2]])
        if np.max(np.abs(normal[:2])) > 0.2 :
            done = True


        return state, reward, done, {}

    def PlotReward(self) :

        c = range(len(self.percentages))
        plt.plot(self.percentMean, c= 'b', alpha = 0.8)
        plt.fill_between(c, np.array(self.percentMean)+np.array(self.percentStd), np.array(self.percentMean)-np.array(self.percentStd), color='g', alpha=0.3, label='Objective 1')

        plt.grid()
        plt.savefig('rews.png')
        plt.close()
