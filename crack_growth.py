from dolfin import *
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
ffc_options = {"optimize": True, \
                "eliminate_zeros": True, \
                "precompute_basis_const": True, \
                "precompute_ip_const": True}   
parameters["form_compiler"]["quadrature_degree"] = 1
set_log_active(False)
mesh = RectangleMesh(Point(-0.5,-0.5),Point(0.5,0.5), 20, 20)
cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.25:
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
    if abs(p[1]) < 0.1:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)


W = VectorFunctionSpace(mesh, 'CG', 1)
p, q, dp = Function(W), TestFunction(W), TrialFunction(W)
u, v, du = Function(W), TestFunction(W), TrialFunction(W)

unew, pnew, uold, pold = Function(W), Function(W), Function(W), Function(W) 

top = CompiledSubDomain("near(x[1], 0.5) && on_boundary")
bot = CompiledSubDomain("near(x[1], -0.5) && on_boundary")
load = Expression("t", t = 0, degree=1)
bcbot= DirichletBC(W, Constant((0.0,0.0)), bot)
bctop = DirichletBC(W.sub(1), load, top)
bc_u = [bcbot, bctop]
bc_phi = []

class InitialCondition(UserExpression):
    def eval_cell(self, value, x, ufl_cell):      
        if abs(x[1]) < 10e-03 and x[0] <= 0:
            value[0] = 0
            value[1] = 1 
        else:
            value[0] = 0
            value[1] = 0          
    def value_shape(self):
        return (2,)

pold.interpolate(InitialCondition())
p.interpolate(InitialCondition())
pnew.interpolate(InitialCondition())

Gc, l, lmbda, mu, eta_eps =  1.721/ (100e3), 0.015, 0, 1, 1.e-3
Cr = 1.e-3
def energy(alpha1, alpha0, beta):    
    Energy = mu/2 *(alpha0**2 + alpha1**2 +beta**2 -2)+ h(alpha0*alpha1)                                                                                                                                                                                   
    return Energy

def W0(u):
    I = Identity(len(u))
    F = variable(I + grad(u))   
    C = F.T*F
    C00, C01, C10, C11 = C[0,0], C[0,1], C[1,0], C[1,1]
    alpha1 =  C00**(0.5) 
    beta =  C01 / alpha1 
    alpha0 = (C11 - beta*beta)**(0.5) 
    E = energy(alpha1, alpha0, beta)
    stress = diff(E, F) 
    return [E, stress]


                                                                                                                                                                                              
def W1(u,d): 
    I = Identity(len(u))
    F = variable (I + grad(u) ) 
    d= variable(d)           
    n1 ,n2 =  d[0]/(sqrt(dot(d,d))) ,  d[1]/(sqrt(dot(d,d)))
    a1, a2 = F[0,0]*n2 - F[0,1]*n1 , F[1,0]*n2 - F[1,1]*n1
    alpha1 =  sqrt(a1**2 + a2**2) 
    alpha0 =  (det(F))/sqrt(a1**2 + a2**2) 
    alpha0_s =   ( ((lmbda*alpha1)**2 +4*mu*lmbda*(alpha1**2)+ 4*mu**2)** (0.5) +\
        lmbda*alpha1)  / (2*(lmbda*(alpha1**2) +mu)) 
    E =   conditional(lt(alpha0, alpha0_s),  energy(alpha1, alpha0, 0), \
          energy(alpha1, alpha0_s, 0) )                            
    stress, dE_dd = diff(E, F) , diff(E, d) 
    return [E, stress, dE_dd]
   
def h(J):
    return (lmbda/2)*(J-1)**2 -mu*ln(J)

def total_energy(u,d):
    E = ((1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 + eta_eps)*W0(u)[0] +\
        (1-(1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 )*\
        conditional(lt(dot(d,d), Cr),0.0, W1(u,d)[0]) +\
        Gc* ( dot(d,d)/(2*l) + (l/2)*inner(grad(d), grad(d)) )              
    return E

def elastic_energy(u,d):
    E = ((1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 + eta_eps)*W0(u)[0] +\
        (1-(1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 )*\
        conditional(lt(dot(d,d), Cr),0.0, W1(u,d)[0])              
    return E
        
Pi1 = total_energy(u, pold) * dx          					    
Pi2 = total_energy(unew, p) * dx 

E_du = derivative(Pi1, u, v)   
E_phi = derivative(Pi2, p, q)
 
J_phi  = derivative(E_phi, p, dp)
J_u = derivative(E_du, u, du)    
p_disp = NonlinearVariationalProblem(E_du, u, bc_u, J_u)
p_phi = NonlinearVariationalProblem(E_phi, p, bc_phi ,J_phi)
solver_disp = NonlinearVariationalSolver(p_disp)
solver_phi = NonlinearVariationalSolver(p_phi)

prm1 = solver_disp.parameters
prm1['newton_solver']['maximum_iterations'] = 1000
prm2 = solver_phi.parameters
prm2['newton_solver']['maximum_iterations'] = 1000

def preventHeal(pold, pnew):  #conserves the direction of old crack
    pold_nodal_values = pold.vector()
    pold_array = pold_nodal_values.get_local()
    pnew_nodal_values = pnew.vector() 
    pnew_array = pnew_nodal_values.get_local()
    for i in range(0, len(pold_array), 2):
        pold_mag = sqrt( (pold_array[i])**2 +(pold_array[i+1])**2 )
        pnew_mag = sqrt( (pnew_array[i])**2 +(pnew_array[i+1])**2 )
        if pold_mag > 0.95:
            pnew_array[i], pnew_array[i+1] =  pold_array[i]/pold_mag, \
                pold_array[i+1]/pold_mag        
    pnew3 = Function(W)        
    pnew3.vector()[:] = pnew_array[:]
    return pnew3

def stress_form(u,d):
    E = ((1- conditional(lt(dot(d,d), Cr), 0, sqrt(dot(d,d))))**2 + eta_eps)*W0(u)[1] +\
        (1-(1- conditional(lt(dot(d,d), Cr),0, sqrt(dot(d,d))))**2 )*\
        conditional(lt(dot(d,d), Cr), stress_null, W1(u,d)[1])              
    return E

TS = TensorFunctionSpace(mesh, "CG", 1)
stress = Function(TS)
stress_null = Function(TS)

V = FunctionSpace(mesh, 'DG', 0)
energy_total = Function(V)
energy_elastic = Function(V)

CrackVector_file = File ("./Result/crack_vector.pvd")
Displacement_file = File ("./Result/displacement.pvd")   
TotalEnergy_file = File ("./Result/total_energy.pvd")
ElasticEnergy_file = File ("./Result/elastic_energy.pvd")
Stress_file = File("./Result/stress.pvd")

t = 0
u_r = 0.003
deltaT  = 0.1
tol = 1e-3

while t<= 1.8: 
    if t>= 1.45:
        deltaT = 1.e-2  
    t += deltaT
    load.t=t*u_r
    iter = 0
    err = 1
    while err > tol:
        iter += 1
        solver_disp.solve()
        unew.assign(u) 
        solver_phi.solve()
        p_new = preventHeal(pold, p)
        pnew.assign(p_new)  
        err_u = errornorm(unew,uold,norm_type = 'l2',mesh = None)
        err_phi = errornorm(pnew,pold,norm_type = 'l2',mesh = None)
        err = max(err_u,err_phi)
        uold.assign(unew)
        pold.assign(pnew)
        if err <= tol:
            print ('Iterations:', iter, ', Total time', t)
            StoringDisplacement_file = File ("./Result/saved_u"+str(t)+ ".xml")
            StoringCrack_file = File ("./Result/saved_p"+str(t)+ ".xml")     
            pnew.rename("d", "crack_vector")
            CrackVector_file << pnew
            Displacement_file << unew
            StoringCrack_file << pnew
            StoringDisplacement_file << unew    
            energy_total = project( total_energy(unew, pnew) ,V)
            energy_total.rename("energy_total", "energy_total")
            TotalEnergy_file << energy_total
            
            energy_elastic = project( elastic_energy(unew, pnew) ,V)
            energy_elastic.rename("energy_elastic", "energy_elastic")
            ElasticEnergy_file << energy_elastic
            
            stress = project( stress_form(unew, pnew), TS) 
            stress.rename("stress", "stress")
            Stress_file << stress
print ('Simulation completed')
