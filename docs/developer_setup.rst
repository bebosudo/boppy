===============
Developer setup
===============

First of all, make sure to run the latest Python 3 version available; on linux, search for it in your package manager (``apt search python`` in ubuntu); on windows, visit https://www.python.org; on mac, you usually use `brew <https://brew.sh/>`_ to install packages.


Virtualenv Setup (first time only)
==================================

Setup a Python `virtualenv <https://virtualenv.pypa.io/en/stable/>`_, which is sort of a sandbox/virtual machine of python packages (or, if you want to perform some more setup in exchange for a better usage, see `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_).

Install virtualenv via pip::

    $ pip install virtualenv

Test your installation::

    $ virtualenv --version

Create a virtual environment for a project::

    $ cd /somewhere/
    $ virtualenv poppyenv

This creates a new directory where a copy of the python3 executable is stored. This is NOT the project directory, where the code is stored (which we are going to illustrate in a second), this is a sandbox containing the python interpreter and the library dependencies needed to run the code.


Virtualenv Usage
================

Once the *virtualenv* has been created, you need to run a configurator script every time you want to use that virtual environment.

On Linux/Mac::

    $ source /somewhere/poppyenv/bin/activate

On Windows::

    > \somewhere\poppyenv\Scripts\activate

This should add a label before your normal prompt label::

    (poppyenv) [user@system]$


To deactivate the virtualenv and return back to the normal python interpreter and libraries, just type ``deactivate`` at the command prompt.

To test once more that the virtualenv is enabled, type ``which python`` (Linux/Mac?) or ``where python`` (Windows): it should return the path to the python3 executable that would be executed when launching python, and it should point to ``/somewhere/poppyenv/bin/python`` (``\somewhere\poppyenv\Scripts\python`` on Windows).


Libraries installation
======================

The code in this repo relies on some other code in order to work properly: these "other codes" are called *dependencies*.
The dependencies we have so far are only other python libraries, so we can handle them using the virtualenv we just set up.

All the dependencies needed are stored in a file called ``requirements.txt`` in the main directory of this repository, and we can install them with (activate the virtualenv first)::

    $ pip install -r requirements.txt

Done!


Run the tests
=============

In this project we use ``nosetest`` to run tests. It automatically finds the directories containing tests and all the test files. A small wrapper is available in the main directory of the project; test it with (Linux/Mac)::

    $ ./run_tests

Or (Windows)::

    > python run_tests

It should output something like::

    $ ./run_tests
    .....
    ----------------------------------------------------------------------
    Ran 5 tests in 0.075s

    OK

Remember to always test that all the tests pass before submitting your changes to the remote repository.


Development strategies
========================

In this project, stick to the Python `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ Style Guide, with the exception of the maximum line length set at 100 chars, for both code and docstrings.

When developing a new feature in this project, try to follow the `Test Driven Development <https://en.wikipedia.org/wiki/Test-driven_development>`_ strategy:

1. Choose a specific feature you want to add, and how the user should use it (this is called a *requirement*);
2. Write *test cases* that your feature should pass once developed; a test case asserts that a certain input produces a specific output. When writing tests, "the more, the merrier".
3. Run the test suite: it should fail (only) on the test cases you just wrote;
4. Program your new feature;
5. Repeat steps 3 to 4 until all the tests pass.


Editor setup
=================

In case you want a suggestion on an editor to use to develop in Python, try `Sublime Text <https://www.sublimetext.com/>`_ editor; unfortunately it's not open-source, but it's free to use (even though it often asks to purchase a license).
Once installed, press ``CTRL + SHIFT + P`` and search ``install Package Control``, which is a sort of central repository of plugins for Sublime Text

Once installed, press again ``CTRL + SHIFT + P`` and search for the package ``Python 3``, which provides syntax highlighting for Python3 code.
Another useful plugin is ``AutoPep8``: it provides an automatic formatter that adjust code to the Python PEP8 Style Guide; in order to trigger it at each file save, go to ``Preferences`` -> ``Package Settings`` -> ``AutoPep8`` -> ``Settings - User`` and paste the following::

    {
        "max-line-length": 99,

        // number of spaces per indent level
        "indent-size": 4,

        "format_on_save": true,

        // Format/Preview menu items only appear for views
        // with syntax from `syntax_list`
        // value is base filename of the .tmLanguage syntax files
        "syntax_list": ["Python", "Python3"],
    }

An optional plugin could be ``reStructuredText Improved``, which gives syntax highlighting to ``.rst`` files, such as this one.

Close and restart and your editor should be up and running.


Git reminder
============

In need of a refresh on git? Here's a `cheatsheet <https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf>`_ with some handy commands.
A quick recap of a usual workflow is:

* ``git pull`` **inside** the poppy directory to update your local repository;
* ``git status`` to check what's the status of your local repo, when you are ready to submit your changes, then
* ``git add file1 file2 file3`` to add the files you want to be part of your commit (or add them separately), then
* ``git commit -m "Write here a short message explaining your changes"`` to "wrap" all those changes in a new snapshot (called a "commit" in the git *jargon*), and eventually
* ``git push`` to send your snapshot(s) to the remote repo.

A more complete documentation on git can be found `here <https://git-scm.com/doc>`_.


Conflict
--------

*Merge conflict* emergency?? `DON'T PANIC <https://www.youtube.com/watch?v=5ilGGP9BDZs>`_! Usually it all boils down to the same steps::

1. ``git status`` to check what files are involved;
2. open those files with your favorite text editor, and search for lines starting with ``<<<<<<<``;
3. choose which version of the two to keep (yours, or the newest one);
4. ``git add fileX`` to mark the file as "solved";
5. repeat steps 2 to 4 for each file involved in the conflict, and finally
6. ``git commit -m "Message exlaining what kind of conflict you just solved"``.

A more complete example: https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/.
