# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXBUILD           = sphinx-build
SOURCEDIR             = source
BUILDDIR              = build
TEMPLATEDIR           = "${SOURCEDIR}/_templates"

SPHINXOPTS            =
SPHINXAPIDOCOPTS      = -e -t ${TEMPLATEDIR} --no-toc -o $(SOURCEDIR) -f

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	-rm -r "$(BUILDDIR)"

apidoc:
	sphinx-apidoc ${SPHINXAPIDOCOPTS} ../src

build:
	# run the Sphinx build for html
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html: apidoc build

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
