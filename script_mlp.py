from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def genere_exemple_dim1(xmin, xmax, NbEx, sigma):
    x = np.arange(xmin, xmax, (xmax - xmin) / NbEx)
    y = np.sin(-np.pi + 2 * x * np.pi) + np.random.normal(
        loc=0, scale=sigma, size=x.size
    )
    return x.reshape(-1, 1), y


def getMSE(x, y, reg):
    return sum(pow(reg.predict(x) - y, 2)) / x.shape[0]


def plot_model(Xa, Ya, Xt, Yt, reg, nameFig):
    Ypred = reg.predict(Xt)
    plt.plot(Xa[:, 1], Ya, "*r")
    plt.plot(Xt[:, 1], Yt, "-b")
    plt.plot(Xt[:, 1], Ypred, "-r")
    plt.grid()
    plt.savefig(nameFig + ".jpg", dpi=200)
    plt.close()


def plot_error_profile(L_error_app, L_error_test, nameFig):
    plt.plot(range(1, len(L_error_app) + 1), L_error_app, "-r")
    plt.plot(range(1, len(L_error_test) + 1), L_error_test, "-b")
    plt.grid()
    plt.savefig(nameFig + ".jpg", dpi=200)
    plt.close()


def plot_confusion(Xt, Yt, reg, nameFig):
    plt.plot(Yt, reg.predict(Xt), ".b")
    plt.plot(Yt, Yt, "-r")
    plt.savefig(nameFig + ".jpg", dpi=200)
    plt.close()


def open_file(file_name) -> np.array:
    with open(file_name, "r") as file:
        file.readline()
        data = np.loadtxt(file, delimiter=";")
    return data


def sinus_cardinal(vect: np.array = None) -> float:
    A = np.array([[1, 1], [-2, 1]])
    b = np.array([0.2, -0.3])
    x = -np.pi + 2 * vect * np.pi
    z = A.dot(x + b)
    h = np.sqrt(np.transpose(z).dot(z))
    if np.abs(h) < 0.001:
        return 1
    else:
        return np.sin(h) / h


def plot_surf(figname, regr=None, show=False, save=False):
    step_v = 0.005

    x1v = np.arange(0, 1, step_v)
    x2v = np.arange(0, 1, step_v)
    Xv, Yv = np.meshgrid(x1v, x2v)

    R = np.zeros(Xv.shape)
    for i, x1 in enumerate(x1v):
        for j, x2 in enumerate(x2v):
            if not regr:
                R[i, j] = sinus_cardinal(np.array([x1, x2]))
            else:
                R[i, j] = regr.predict(np.array([[x1, x2]]))[0]

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # Plot the surface
    if regr:
        label = "model"
    else:
        label = "sinc 2D"
    surf = ax.plot_surface(
        Xv, Yv, R, cmap=cm.coolwarm, linewidth=0, antialiased=False, label=label
    )

    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    plt.legend()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if save:
        plt.savefig(figname, dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_app_test(Xapp, Yapp, Xtest, Ytest, show=False, save=False, figname=None):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # Plot the surface

    ax.scatter(Xtest[:, 0], Xtest[:, 1], Ytest, color="b", label="Test")
    ax.scatter(Xapp[:, 0], Xapp[:, 1], Yapp + 0.001, color="r", label="Apprentissage")

    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    plt.legend()
    if save:
        if figname == None:
            raise ValueError("No figname given")
        plt.savefig(figname + ".jpg", dpi=300)
    if show:
        plt.show()
    plt.close()


def main():
    Xapp = open_file("sinc_dim2_input.csv")
    Yapp = np.array([sinus_cardinal(Xapp[i]) for i in range(np.shape(Xapp)[0])])
    plot_surf("Sin_card_plot", save=True)

    num_points = 40
    grid_line = np.linspace(0, 1, num_points)

    Xtest = np.zeros((num_points**2, 2))

    for i in range(num_points):
        for j in range(num_points):
            Xtest[i * num_points + j] = [grid_line[i], grid_line[j]]

    Ytest = np.array([sinus_cardinal(Xtest[i]) for i in range(np.shape(Xtest)[0])])

    plot_app_test(
        Xapp=Xapp,
        Yapp=Yapp,
        Xtest=Xtest,
        Ytest=Ytest,
        show=True,
        save=True,
        figname="app_test_MLP",
    )

    L_error_app = []
    L_error_test = []
    reg_list = []
    hidden_layer_max = 30

    for i in range(1, hidden_layer_max + 1):
        print("hidden_layer size = ", i)

        # Création du modèle linéaire
        reg = MLPRegressor(
            hidden_layer_sizes=(i,),
            max_iter=2000,
            activation="tanh",
            solver="lbfgs",
            tol=0.0001,
        ).fit(Xapp, Yapp)
        reg_list.append(reg)

        # Estimation des erreurs d'apprentissage et de test
        L_error_app.append(getMSE(Xapp, Yapp, reg))
        L_error_test.append(getMSE(Xtest, Ytest, reg))

        # plot du model de degré i
        # plot_model(Xa, yapp, Xt, ytest, reg, "Model_%02d" % i)

    best = np.argmin(L_error_test) + 1
    best_reg = reg_list[best - 1]
    print("Meilleur model -> hidden_size =", best)
    plot_error_profile(L_error_app, L_error_test, "Profil_Err_App_Test_MLP")

    plot_surf("Meilleur model MLP", regr=best_reg, save=True)

    plot_confusion(Xtest, Ytest, best_reg, "Confusion_MLP")


if __name__ == "__main__":
    main()
