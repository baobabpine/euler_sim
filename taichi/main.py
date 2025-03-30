import taichi as ti

import argparse
arch = ti.cpu
ti.init(arch=arch)

# gui system using taichi-ggui:
# https://docs.taichi.graphics/zh-Hans/docs/lang/articles/misc/g)ui
# imgSize = 512
imgSize = 1024

clothWid = 4.0
clothHgt = 4.0
clothResX = 31
# 31
N = clothResX
# NE = 2 * clothResX * (clothResX + 1) + 2 * clothResX * clothResX
NE = (N - 1)*(3*N-1)

# num_triangles = clothResX * clothResX * 2
num_triangles = (clothResX-1) * (clothResX-1) * 2
num_vertices = clothResX * clothResX
num_spring = NE
indices = ti.field(int, num_triangles * 3)
edgelist = ti.Vector.field(2, dtype=ti.i32, shape=num_triangles * 3)
edge = []
tem = ti.field(int, 3)
idx = ti.field(int, num_triangles * 3)
# vertices = ti.Vector.field(3, float, (clothResX + 1) * (clothResX + 1))
vertices = ti.Vector.field(3, float, (clothResX ) * (clothResX ))

pos_pre = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX , clothResX ))
pos = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX , clothResX))
vel = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX , clothResX ))
F = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX , clothResX ))
J = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(clothResX , clothResX ))
fext = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX , clothResX ))

spring = ti.Vector.field(2, ti.i32, NE)
pre_length = ti.field(ti.f32, NE)

KsStruct = ti.field(dtype=ti.f32, shape=())
KdStruct = ti.field(dtype=ti.f32, shape=())
KsShear = ti.field(dtype=ti.f32, shape=())
KdShear = ti.field(dtype=ti.f32, shape=())
KsBend = ti.field(dtype=ti.f32, shape=())
KdBend = ti.field(dtype=ti.f32, shape=())

# dm = ti.ndarray(ti.f32, 3 * num_spring)
dm = ti.linalg.SparseMatrixBuilder(3 * num_spring, 1, max_num_triplets=1000000)
ym = ti.ndarray(ti.f32, 3 * num_vertices)
mhl = ti.linalg.SparseMatrixBuilder(3 * num_vertices, 3 * num_vertices, max_num_triplets=1000000)
jm = ti.linalg.SparseMatrixBuilder(3 * num_vertices, 3 * num_spring, max_num_triplets=1000000)
res_right = ti.ndarray(ti.f32, 3*num_vertices)


def comp(f1, f2):
    if f1[0] < f2[0]:
        return True
    if f1[0] > f2[0]:
        return False
    if f1[1] < f2[1]:
        return True
    if f1[1] > f2[1]:
        return False
    return False


@ti.func
def getNextNeighborKs(n):
    ks = 0.0
    if (n < 4):
        ks = KsStruct
    else:
        if (n < 8):
            ks = KsShear
        if (n < 12):
            ks = KsBend
    return ks


@ti.func
def getNextNeighborKd(n):
    kd = 0.0
    if (n < 4):
        kd = KdStruct
    else:
        if (n < 8):
            kd = KdShear
        else:
            kd = KdBend
    return kd


@ti.func
def getNextNeighborX(n):
    cood = 0
    if (n == 0) or (n == 4) or (n == 7):
        cood = 1
    if (n == 2) or (n == 5) or (n == 6):
        cood = -1
    if (n == 8):
        cood = 2
    if (n == 10):
        cood = -2
    return cood


@ti.func
def getNextNeighborY(n):
    cood = 0
    if (n == 1) or (n == 4) or (n == 5):
        cood = -1
    if (n == 3) or (n == 6) or (n == 7):
        cood = 1
    if (n == 9):
        cood = -2
    if (n == 11):
        cood = 2
    return cood


@ti.func
def get_length3(v):
    return ti.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


@ti.func
def get_length2(v):
    return ti.sqrt(v.x * v.x + v.y * v.y)


@ti.func
def SolveConjugateGradient(A, x, b):  # implicit Euler block matrix solve
    r = b - A @ x
    d = r
    q = ti.Vector([0.0, 0.0, 0.0])
    alpha_new = 0.0
    alpha = 0.0
    beta = 0.0
    delta_old = 0.0
    delta_new = r.dot(r)
    delta0 = delta_new

    for i in range(0, 16):
        q = A @ d
        alpha = delta_new / d.dot(q)
        x = x + alpha * d
        r = r - alpha * q
        delta_old = delta_new
        delta_new = r.dot(r)
        beta = delta_new / delta_old
        d = r + beta * d
        i += 1
        if delta_new < 0.0000001:
            break
    return x


@ti.kernel
def reset_cloth():  # initial
    for i, j in pos:
        pos[i, j] = ti.Vector(
            [clothWid * (i / clothResX) - clothWid / 2.0, 5.0, clothHgt * (j / clothResX) - clothHgt / 2.0])
        pos_pre[i, j] = pos[i, j]
        vel[i, j] = ti.Vector([0.0, 0.0, 0.0])
        F[i, j] = ti.Vector([0.0, 0.0, 0.0])
        # print("i=", i, "j=", j, "pos=", pos[i, j][0], ",", pos[i, j][1], ",", pos[i, j][2], "\n")

        if i < clothResX-1 and j < clothResX-1:
            tri_id = ((i * (clothResX - 1)) + j) * 2
            indices[tri_id * 3 + 2] = i * clothResX + j
            indices[tri_id * 3 + 1] = (i + 1) * clothResX + j
            indices[tri_id * 3 + 0] = i * clothResX + (j + 1)

            tri_id += 1
            indices[tri_id * 3 + 2] = (i + 1) * clothResX + j + 1
            indices[tri_id * 3 + 1] = i * clothResX + (j + 1)
            indices[tri_id * 3 + 0] = (i + 1) * clothResX + j
            # print("i=", i, "j=", j, "tri_id=", ((i * (clothResX - 1)) + j) * 2, ",", i * clothResX + j, ",", (i + 1) * clothResX + j, ",",  i * clothResX + (j + 1), "\n")
            # print("i=", i, "j=", j, "tri_id+1=", tri_id, ",", (i + 1) * clothResX + j + 1, ",",
                  # i * clothResX + (j + 1), ",", (i + 1) * clothResX + j, "\n")
    ball_centers[0] = ti.Vector([0.0, 3.0, 0.0])
    ball_centers0[0] = ti.Vector([0.0, 0.0, 0.0])
    ball_radius[0] = 1.0
    deltaT[0] = 1.2


def init_edges():
    for i in range(0, num_triangles):
        tem[0] = indices[3 * i]
        tem[1] = indices[3 * i + 1]
        tem[2] = indices[3 * i + 2]
        ti.algorithms.parallel_sort(tem, values=None)
        edge.append((tem[0], tem[2]))
        edge.append((tem[1], tem[2]))
        edge.append((tem[0], tem[1]))

    edge.sort()

    itr = 0
    if edge[0][0] != edge[1][0] or edge[0][1] != edge[1][1]:
        idx[itr] = 0
        itr = 1

    for i in range(1, num_triangles*3):
        if edge[i][0] != edge[i - 1][0] or edge[i][1] != edge[i - 1][1]:
            idx[itr] = i
            itr = itr + 1
    global edgelist
    edgelist = ti.Vector.field(2, dtype=ti.i32, shape=itr)
    global NE
    NE = itr
    for i in range(0, itr):
        edgelist[i][0] = edge[idx[i]][0]
        edgelist[i][1] = edge[idx[i]][1]
        idx1 = edgelist[i][0]
        idx2 = edgelist[i][1]
        coord0 = ti.Vector([int(idx1 / clothResX), idx1 % clothResX])
        coord1 = ti.Vector([int(idx2 / clothResX), idx2 % clothResX])
        tmp = 0.0
        p = pos[int(idx1 / clothResX), idx1 % clothResX]-pos[int(idx2 / clothResX), idx2 % clothResX]
        for k in range(0, 3):
            tmp += p[k]*p[k]
        tmp = ti.sqrt(tmp)
        pre_length[i] = tmp


@ti.func
def compute_force(coord, jacobian):
    p1 = pos[coord]
    v1 = vel[coord]
    F[coord] = gravity * mass + vel[coord] * damping  # gravity and friction
    E = ti.Matrix.identity(ti.f32, 3)
    J[coord] = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    for k in range(0, 12):
        ks = getNextNeighborKs(k)
        kd = getNextNeighborKd(k)

        coord_offcet = ti.Vector([getNextNeighborX(k), getNextNeighborY(k)])
        coord_neigh = coord + coord_offcet

        if (coord_neigh.x >= 0) and (coord_neigh.x < clothResX) and (coord_neigh.y >= 0) and (
                coord_neigh.y < clothResX):

            rest_length = get_length2(coord_offcet * ti.Vector([clothWid / clothResX, clothHgt / clothResX]))

            p2 = pos[coord_neigh]
            v2 = (p2 - pos_pre[coord_neigh]) / deltaT[0]

            deltaP = p1 - p2
            deltaV = v1 - v2
            dist = get_length3(deltaP)

            if jacobian > 0:
                dist2 = dist * dist
                lo_l = rest_length / dist
                J[coord] += ks * (lo_l * (E - deltaP.outer_product(deltaP) / dist2) - E)

            leftTerm = -ks * (dist - rest_length)
            rightTerm = kd * (deltaV.dot(deltaP) / dist)
            F[coord] += deltaP.normalized() * (leftTerm + rightTerm)  # spring force


@ti.func
def collision(coord):  # SDF
    # collision
    if (pos[coord].y < 0):
        pos[coord].y = 0

    offcet = pos[coord] - ball_centers[0]
    dist = get_length3(offcet)
    if (dist < ball_radius[0]):
        delta0 = (ball_radius[0] - dist)
        pos[coord] += offcet.normalized() * delta0
        pos_pre[coord] = pos[coord]
        vel[coord] = ti.Vector([0.0, 0.0, 0.0])
    if (pos[coord].y < 1):
        pos[coord].y = 0.1
        vel[coord] = ti.Vector([0.0, 0.0, 0.0])
        F[coord] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def integrator_verlet():  # semi- implicit Euler
    for i, j in pos:
        coord = ti.Vector([i, j])
        compute_force(coord, 0)
        index = j * (clothResX + 1) + i

        if (index != 0) and (index != clothResX):
            collision(coord)

            acc = F[coord] / mass
            tmp = pos[coord]
            pos[coord] = pos[coord] * 2.0 - pos_pre[coord] + acc * deltaT[0] * deltaT[0]
            vel[coord] = (pos[coord] - pos_pre[coord]) / deltaT[0]
            pos_pre[coord] = tmp


@ti.kernel
def integrator_explicit():  # semi - implicit Euler
    for i, j in pos:
        coord = ti.Vector([i, j])
        compute_force(coord, 0)
        index = j * (clothResX + 1) + i

        if (index != 0) and (index != clothResX):
            collision(coord)

            tmp = pos[coord]
            acc = F[coord] / mass
            pos[coord] += vel[coord] * deltaT[0]
            vel[coord] += acc * deltaT[0]
            pos_pre[coord] = tmp


@ti.kernel
def integrator_implicit():  # implicit Euler
    for i, j in pos:
        coord = ti.Vector([i, j])
        compute_force(coord, 1)
        index = j * (clothResX + 1) + i

        if (index != 0) and (index != clothResX):
            collision(coord)
            tmp = pos[coord]
            M = ti.Matrix.identity(ti.f32, 3) * mass
            A = M - deltaT[0] * deltaT[0] * J[coord]
            b = M @ vel[coord] + deltaT[0] * F[coord]

            vel[coord] = SolveConjugateGradient(A, ti.Vector([0.0, 0.0, 0.0]), b)
            pos[coord] += vel[coord] * deltaT[0]
            pos_pre[coord] = tmp


@ti.kernel
def update_verts():
    for i, j in ti.ndrange(clothResX, clothResX):
        vertices[i * clothResX + j] = pos[i, j]


@ti.func
def compute_f(coord):
    F[coord] = gravity * mass + vel[coord] * damping


@ti.kernel
def assemble_mhl(mhl: ti.types.sparse_matrix_builder()):  # m + h * h * k
    for i in range(0, NE):
        idx1 = edgelist[i][0]
        idx2 = edgelist[i][1]
        for j in range(0, 3):
            mhl[3 * idx1 + j, 3 * idx1 + j] += KsStruct * deltaT[0] * deltaT[0]
            mhl[3 * idx1 + j, 3 * idx2 + j] -= KsStruct * deltaT[0] * deltaT[0]
            mhl[3 * idx2 + j, 3 * idx1 + j] -= KsStruct * deltaT[0] * deltaT[0]
            mhl[3 * idx2 + j, 3 * idx2 + j] += KsStruct * deltaT[0] * deltaT[0]
    for i in range(0, num_vertices):
        for j in range(0, 3):
            mhl[3 * i + j, 3 * i + j] += mass


@ti.kernel
def assemble_dm(testd: ti.types.sparse_matrix_builder()):  # distance
    for i in range(0, NE):
        idx1 = edgelist[i][0]
        idx2 = edgelist[i][1]
        coord0 = ti.Vector([int(idx1 / clothResX), idx1 % clothResX])
        coord1 = ti.Vector([int(idx2 / clothResX), idx2 % clothResX])
        p = pos[coord0] - pos[coord1]
        tmp = 0.0
        for k in range(0, 3):
            tmp += (pos[coord0][k] - pos[coord1][k]) * (pos[coord0][k] - pos[coord1][k])
        tmp = ti.sqrt(tmp)
        pn = tmp
        for j in range(0, 3):
            testd[3*i+j, 0] += pre_length[i] * p[j] * (1 / pn)


@ti.kernel
def assemble_d2(td: ti.types.sparse_matrix_builder(), x: ti.types.ndarray()):
    for i in range(0, NE):
        idx1 = edgelist[i][0]
        idx2 = edgelist[i][1]
        tmp = 0.0
        for j in range(0, 3):
            tmp += (x[3 * idx1 + j] - x[3 * idx2 + j]) * (x[3 * idx1 + j] - x[3 * idx2 + j])
        tmp = ti.sqrt(tmp)
        for j in range(0, 3):
            p = x[3 * idx1 + j] - x[3 * idx2 + j]
            td[3 * i + j, 0] += pre_length[i] * p * (1 / tmp)


@ti.kernel
def assemble_jm(testj: ti.types.sparse_matrix_builder()):
    for i in range(0, NE):
        idx1 = edgelist[i][0]
        idx2 = edgelist[i][1]
        for j in range(0, 3):
            testj[3 * idx1 + j, 3 * i + j] += KsStruct
            testj[3 * idx2 + j, 3 * i + j] -= KsStruct


@ti.kernel
def assemble_ym(testy: ti.types.ndarray()):
    for i in range(0, num_vertices):
        for j in range(0, 3):
            coord = ti.Vector([int(i / clothResX), i % clothResX])
            testy[3*i+j] = pos[coord][j] + deltaT[0] * vel[coord][j] + deltaT[0] * deltaT[0] * (1/mass) * F[coord][j]


def fast_method():  # fast simulation
    assemble_dm(dm)
    dmb = dm.build()
    assemble_ym(ym)
    assemble_mhl(mhl)
    mhlb = mhl.build()
    assemble_jm(jm)
    jmb = jm.build()
    for testi in range(0, 10):
        right = jmb @ dmb  # global
        for i in range(num_vertices):
            for j in range(3):
                res_right[3*i+j] = deltaT[0] * deltaT[0] * right[3*i+j, 0] + mass * ym[3*i+j]

        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(mhlb)
        solver.factorize(mhlb)
        x = solver.solve(res_right)  # local
        assemble_d2(dm, x)
        dmb = dm.build()

        # for scr in range(0, num_vertices):
        #     print(scr, ": ", x[3*scr], ", ", x[3*scr+1], ", ", x[3*scr+2], ", ")

    for i in range(0, num_vertices):
        for j in range(0, 3):
            y = int(i / clothResX)
            z = i % clothResX
            # coord = ti.Vector([int(i / (clothResX + 1)), i % (clothResX + 1)])
            vel[y, z][j] = (x[3 * i + j] - pos[y, z][j]) / deltaT[0]
            pos[y, z][j] = x[3 * i + j]
            # print("i=%d j=%d [%d, %d] pos=%f vel=%f\n"%(i, j, y, z, pos[y, z][j], vel[y, z][j]))
            # vel[coord][j] = (x[3*i+j]-pos[coord][j])/deltaT[0]
            # pos[coord][j] = x[3*i+j]


@ti.kernel
def init_collision():
    for i, j in pos:
        coord = ti.Vector([i, j])
        # compute_force(coord, 1)
        compute_f(coord)
        index = i * clothResX + j

        if (index != 0) and (index != clothResX):
            collision(coord)


gravity = ti.Vector([0.0, -0.05, 0.0])
ball_centers = ti.Vector.field(3, float, 1)
ball_centers0 = ti.Vector.field(3, float, 1)
ball_radius = ti.field(float, shape=(1))
deltaT = ti.field(float, shape=(1))

mass = 1.0
damping = -0.0125

KsStruct = 10
KdStruct = -0.25
KsShear = 10
KdShear = -0.25
KsBend = 10
KdBend = -0.25

gui = ti.ui.Window('Cloth', (imgSize, imgSize), vsync=True)
canvas = gui.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(5.0, 3.0, 5.0)
camera.lookat(0.0, 3.0, 0.0)
camera.up(0.0, 1.0, 0.0)
mode = 3
reset_cloth()
init_edges()
frame = 0

while gui.running:
    for i in range(0, 3):
        if mode == 0:
            integrator_explicit()
        if mode == 1:
            integrator_implicit()
        if mode == 2:
            integrator_verlet()
        if mode == 3:
            init_collision()
            # print("success 2\n")
            fast_method()
            # print("success 3\n")

    update_verts()
    scene.mesh(vertices, indices=indices, color=(0.5, 0.5, 0.5), two_sided=False, show_wireframe=False)
    scene.particles(ball_centers, radius=0.95, color=(1.0, 0, 0))

    scene.point_light(pos=(10.0, 10.0, 0.0), color=(1.0, 1.0, 1.0))
    camera.track_user_inputs(gui, movement_speed=0.03, hold_key=ti.ui.LMB)
    scene.set_camera(camera)

    canvas.scene(scene)
    gui.show()
