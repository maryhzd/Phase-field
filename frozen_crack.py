from dolfin import *
from dolfin_adjoint import *
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
ffc_options = {"optimize": True, \
                "eliminate_zeros": True, \
                "precompute_basis_const": True, \
                "precompute_ip_const": True}   
parameters["form_compiler"]["quadrature_degree"] = 5
################ defining mesh  ################
mesh = RectangleMesh(Point(-0.5,-0.5),Point(0.5,0.5), 4, 4)
cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.3:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)
cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.15:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.15:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.15:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.125:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.125:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

################ defining space functions  ################
V = VectorFunctionSpace(mesh, "CG", 1)
dy, dd = TrialFunction(V), TrialFunction(V)       
vy, vd = TestFunction(V), TestFunction(V)          
y, d = Function(V), Function(V)

def left(x, on_boundary):
    return abs(x[0]+0.5) < DOLFIN_EPS and on_boundary
def right(x, on_boundary):
    return abs(x[0] -0.5) <  DOLFIN_EPS and on_boundary
def bottom(x, on_boundary):
    return abs (x[1]+0.5) <  DOLFIN_EPS  and on_boundary         
def top(x, on_boundary):
    return abs(x[1]-0.5) < DOLFIN_EPS and on_boundary

F11= 1
F12= 0
F21= 0
F22= 1.4

y_BC = Expression(("F11*x[0]+F12*x[1]", "F21*x[0]+F22*x[1]"), F11=F11,F12=F12,F21=F21,F22=F22, degree=1)
BC1, BC2 = DirichletBC(V, y_BC, top), DirichletBC(V, y_BC, bottom)
BC3, BC4 = DirichletBC(V, y_BC, left) , DirichletBC(V, y_BC, right)                              
BCS = [BC1, BC2, BC3, BC4]
y_assign = Expression(("F11*x[0]+F12*x[1]+1.e-5*0.05* sin(pi*x[0])*sin(pi*x[1])", "F21*x[0]+F22*x[1]+ 1.e-5*0.05* sin(pi*x[0])*sin(pi*x[1])"),\
            F11=F11,F12=F12,F21=F21,F22=F22, degree=1)
y.assign(project(y_assign, V))
y_reference = Expression(("x[0]", "x[1]"), degree=1)

class CircularCrackExpression(UserExpression):
    def eval(self, value, x):       
        a=0.1
        R=1.1 
        if (x[0] /a)**2 + (x[1]/a)**2 <=1:
            value[0] = 0
            value[1] = 1
        elif (x[0] /a)**2 + (x[1]/a)**2  >= (R**2):
            value[0] = 0
            value[1] = 0
        else:
            value[0] =  0 * (R**2 -(x[0] /a)**2 - (x[1]/a)**2) 
            value[1] = 1 *(R**2 -(x[0] /a)**2 - (x[1]/a)**2)         
    def value_shape(self):
        return (2,)

d_initial= CircularCrackExpression(degree=3) 
d.assign(interpolate(d_initial,V))

################ elasticity functions  ################
Gc, l, lmbda, mu, eta_eps =  1.721/ (100e3), 0.015, 233.33e3/ (100e3), 1, 1.e-3
Cr = 1.e-3
def energy(alpha1, alpha0, beta):    
    Energy = mu/2 *(alpha0**2 + alpha1**2 +beta**2 -2)+ h(alpha0*alpha1)                                                                                                                                                                                   
    return Energy

def W0(y):
    F =  grad(y)   
    C = F.T*F
    C00, C01, C10, C11 = C[0,0], C[0,1], C[1,0], C[1,1]
    alpha1 =  C00**(0.5) 
    beta =  C01 / alpha1 
    alpha0 = (C11 - beta*beta)**(0.5) 
    E = energy(alpha1, alpha0, beta) 
    return E

def W1(y,d): 
    F = variable ( grad(y) ) 
    n1 ,n2 =  d[0]/(sqrt(dot(d,d))) ,  d[1]/(sqrt(dot(d,d)))
    a1, a2 = F[0,0]*n2 - F[0,1]*n1 , F[1,0]*n2 - F[1,1]*n1
    alpha1 =  sqrt(a1**2 + a2**2) 
    alpha0 =  (det(F))/sqrt(a1**2 + a2**2) 
    alpha0_s =   ( ((lmbda*alpha1)**2 +4*mu*lmbda*(alpha1**2)+ 4*mu**2)** (0.5) +\
        lmbda*alpha1)  / (2*(lmbda*(alpha1**2) +mu)) 
    E =   conditional(lt(alpha0, alpha0_s),  energy(alpha1, alpha0, 0), \
          energy(alpha1, alpha0_s, 0) ) 
    return E                                                                                                                                                                                             
   
def h(J):
    return (lmbda/2)*(J-1)**2 -mu*ln(J)

def elastic(y, d): 
    E = ((1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 + eta_eps)*W0(y) +\
        (1-(1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2) *\
        conditional(lt(dot(d,d), Cr),0.0, W1(y,d))       
    return E
        
################ minimzng energy function  ################
vy= project(y, V, bcs=BCS) 
J = assemble( (elastic(vy, d))*dx )
J_hat = ReducedFunctional(J, [Control(y)])   
y_opt = minimize(J_hat,  method = "L-BFGS-B", \
                  options = {"gtol": 1e-8, "ftol": 1e-10 ,"maxiter": 3000000 }) 
J_hat(y_opt)  
y = Function(V) 
y = project (y_opt, V, bcs=BCS)

################ storing data  ################           
crack_vector = Function(V)
crack_vector = d
parameters['allow_extrapolation'] = True
d.rename("crack_vector", "crack_vector")
fileD = File("./ResultsDir/crack_vector.pvd");
fileD << crack_vector;

u  = Function(V) 
u = project(y - y_reference, V)
u.rename("u","displacement")
fileU = File("./ResultsDir/displacement.pvd");
fileU << u

FS = FunctionSpace(mesh, "CG", 1)
W_field = project(elastic(y, d), FS)  
W_field.rename("W_field","energy_density") 
fileW = File("./ResultsDir/Energy_Density.pvd")
fileW << W_field

def S0(y): 
    F = variable(grad(y))             
    C = F.T*F
    C00, C01, C10, C11 = C[0,0], C[0,1], C[1,0], C[1,1]
    alpha1 =  C00**(0.5) 
    beta =  C01 / alpha1 
    alpha0 = (C11 - beta*beta)**(0.5) 
    E = energy(alpha1, alpha0, beta)
    stress = diff(E,F)
    return stress

def S1(y,d): 
    F = variable ( grad(y) ) 
    n1 ,n2 =  d[0]/(sqrt(dot(d,d))) ,  d[1]/(sqrt(dot(d,d)))
    a1, a2 = F[0,0]*n2 - F[0,1]*n1 , F[1,0]*n2 - F[1,1]*n1
    alpha1 =  sqrt(a1**2 + a2**2) 
    alpha0 =  (det(F))/sqrt(a1**2 + a2**2) 
    alpha0_s =   ( ((lmbda*alpha1)**2 +4*mu*lmbda*(alpha1**2)+ 4*mu**2)** (0.5) +\
        lmbda*alpha1)  / (2*(lmbda*(alpha1**2) +mu)) 
    E =   conditional(lt(alpha0, alpha0_s),  energy(alpha1, alpha0, 0), \
          energy(alpha1, alpha0_s, 0) )                  
    stress = diff(E,F)                                                                                                                                                                                                
    return stress

def stress_form(y, d): 
    S = ((1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 + eta_eps)*S0(y) +\
        (1-(1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2) *\
        conditional(lt(dot(d,d), Cr), stress_null, S1(y,d))       
    return S

TS = TensorFunctionSpace(mesh, "CG", 1)
stress = Function(TS)
stress_null = Function(TS)

stress = project( stress_form(y, d), TS) 
stress.rename("stress", "stress")
fileS = File("./ResultsDir/stress.pvd")
fileS << stress

e2 = Constant((0, 1))
traction = project(dot(stress,e2), V)
traction.rename("traction", "traction")
fileT = File("./ResultsDir/traction.pvd")
fileT << traction

F = grad(y)
I = Identity(len(u)) 
strain = project ( (F.T * F - I)/2 , TS)
strain.rename("strain", "strain")
fileSt = File("./ResultsDir/strain.pvd")
fileSt << strain

conc_uu = File ("./ResultsDir/saved_u.xml")
conc_ff = File ("./ResultsDir/saved_p.xml")
conc_ff << d
conc_uu << u

parameters['allow_extrapolation'] = False
mesh2 = RectangleMesh.create([Point(-0.5, -0.5),Point(0.5, 0.5)],[30,30],CellType.Type.quadrilateral)
V2 = VectorFunctionSpace(mesh2, "CG", 1)
FS2 = FunctionSpace(mesh2, "CG", 1)
u2 = project(u, V2)
u2.rename("u2","displacement2")
fileUU = File("./ResultsDir/displacement2.pvd");
fileUU << u2

traction2 = interpolate(traction, V2)
traction2.rename("traction2","traction")
filett = File("./ResultsDir/traction2.pvd");
filett << traction2

W_Field = interpolate( W_field , FS2)  
W_Field.rename("W_Field","W_Field") 
fileWW = File("./ResultsDir/W_field2.pvd")
fileWW << W_Field