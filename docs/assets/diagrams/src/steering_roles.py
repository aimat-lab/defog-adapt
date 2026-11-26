"""What a steered run needs, and what of it is optional (Guides, "Steer toward a target").

The main flow is solid: the proposal (a frozen base model with an adapter on it) generates the molecules. Around it, dashed
frames mark what a run can do without: a size model that draws the number of atoms, a Feynman-Kac add-on whose ruler (a
property head) scores each candidate and resamples, and a judge that measures the decoded molecules (RDKit, a better model, or
nothing at all). Each card says what it is, its class, whether it is required and whether it is a file you load. The ruler
steers the run; it is not a measurement.

Build:  python3 $FIGURE_SKILL/scripts/build.py docs/assets/diagrams/src/steering_roles.py [--final]
"""
from figlib import *  # noqa: F401,F403
from vocab import *  # noqa: F401,F403

NAME = 'steering-roles'
SIZE = (440, 218)

# -- layout (pt). Every position derives from these numbers.
ROW_Y, ROW_H = 22, 84                     # the main row: top edge of the cards (where the pills sit) and their height
MID_Y = ROW_Y + ROW_H / 2                 # the line the arrows of the main row sit on
SIZE_X, SIZE_W = 8, 84                    # the size model
PROP_X, PROP_W = 124, 190                 # the proposal
JUDGE_X, JUDGE_W = 346, 86                # the judge
FK_Y, FK_H = 136, 60                      # the Feynman-Kac add-on, below the proposal
LEGEND_Y = 207
# What each card says: (name, class, required). tests/test_diagrams.py ties every class to defog_adapt.core and every
# required/optional mark to the signatures of the samplers, so a rename or a changed default shows up there.
PARTS = {
    'adapter': ('Adapter', 'CrossAttentionAdapter', True),
    'base': ('Base model', 'DeFoGModel', True),
    'ruler': ('Property head', 'PropertyHead', False),
    'size': ('Size model', 'LearnedSizeDistribution', False),
}
NOTE = dict(size=6.3, color=GREY)
CLASS = dict(size=6.0, color=GREY)
ROLE = dict(size=6.5, font='rcm')         # what a part does: generates, steers, measures


def name_class_tag(fig, x, y, name, cls, required, extra='· file'):
    """Three lines at x, the first on the baseline y: the name, the class and the required/optional tag."""
    fig.text(x, y, name, size=7, font='rcm')
    fig.text(x, y + 8, cls, **CLASS)
    tag(fig, x, y + 16, required=required, extra=extra)


def build(fig):
    background(fig)

    # --- the size model (optional): draws the number of atoms from P(n | target)
    name, cls, required = PARTS['size']
    fig.panel(SIZE_X, ROW_Y, SIZE_W, ROW_H, name, fill=PILL_GREY)
    fig.emoji('💧', SIZE_X + 14, ROW_Y + 24, size=9)
    base_y = ROW_Y + 40
    fig.line(SIZE_X + 28, base_y, SIZE_X + 63, base_y, stroke=LABEL_GREY, sw=0.6, name='axis')
    for k, h in enumerate((5, 11, 16, 10, 4)):
        fig.rect(SIZE_X + 30 + 7 * k, base_y - h, 5, h, radius=0.8, fill=GREY, stroke='none', name='size bar')
    fig.text(SIZE_X + SIZE_W / 2, ROW_Y + 50, 'draws the number', justify='center', **NOTE)
    fig.text(SIZE_X + SIZE_W / 2, ROW_Y + 58, 'of atoms', justify='center', **NOTE)
    fig.text(SIZE_X + SIZE_W / 2, ROW_Y + 69, cls, justify='center', **CLASS)
    word = text_width(('required' if required else 'optional') + ' · file', 6.0) + 6
    tag(fig, SIZE_X + SIZE_W / 2 - word / 2, ROW_Y + 78, required=required, extra='· file')
    fig.arrow([(SIZE_X + SIZE_W + 3, MID_Y), (PROP_X - 3, MID_Y)], head=3.0)
    fig.text((SIZE_X + SIZE_W + PROP_X) / 2, MID_Y - 6, 'n atoms', justify='center', **NOTE)

    # --- the proposal (required): an adapter on a frozen base model, both files you load
    fig.card(PROP_X, ROW_Y, PROP_W, ROW_H, name='proposal')
    fig.pill(PROP_X + 8, ROW_Y, 'Proposal', fill=ADAPTER_FILL)
    mx, my, mw, mh = PROP_X + 32, ROW_Y + 32, 40, 34
    frozen_model(fig, mx, my, mw, mh, '')
    ax, a_top = dock_adapter(fig, mx, my, mw, mh)
    fig.emoji('💧', mx - 7, a_top + 7.5, size=9)
    lx = PROP_X + 88                                           # the column of labels
    fig.line(ax + 14, a_top + 7.5, lx - 4, a_top + 7.5, stroke=LABEL_GREY, sw=0.6, name='leader')
    fig.line(mx + mw + 3, my + mh / 2, lx - 4, my + mh / 2, stroke=LABEL_GREY, sw=0.6, name='leader')
    name_class_tag(fig, lx, a_top + 7.5 - 4.5, *PARTS['adapter'])
    name_class_tag(fig, lx, my + mh / 2 - 4.5, *PARTS['base'])
    fig.text(PROP_X + PROP_W / 2, ROW_Y + ROW_H - 6, 'generates the molecules', justify='center', **ROLE)

    # --- the judge (optional): measures the decoded molecules
    fig.arrow([(PROP_X + PROP_W + 3, MID_Y), (JUDGE_X - 3, MID_Y)], head=3.0)
    fig.text((PROP_X + PROP_W + JUDGE_X) / 2, MID_Y - 6, 'decode', justify='center', **NOTE)
    fig.panel(JUDGE_X, ROW_Y, JUDGE_W, ROW_H, 'Judge', fill=PILL_GREY)
    jx = JUDGE_X + JUDGE_W / 2
    fig.text(jx, ROW_Y + 26, 'RDKit', justify='center', size=7, font='rcm')
    fig.text(jx, ROW_Y + 35, 'or a better model,', justify='center', **NOTE)
    fig.text(jx, ROW_Y + 43, 'or no check at all', justify='center', **NOTE)
    fig.text(jx, ROW_Y + 58, 'measures the run', justify='center', **ROLE)
    word = text_width('optional', 6.0) + 6
    tag(fig, jx - word / 2, ROW_Y + 77, required=False)

    # --- the Feynman-Kac add-on (optional): a ruler scores the candidates, the proposal resamples them
    fig.panel(PROP_X, FK_Y, PROP_W, FK_H, 'Feynman-Kac steering', fill=FK_FILL)
    rx, ry, rw, rh = PROP_X + 14, FK_Y + 20, 44, 34
    fig.model(rx, ry, rw, rh, None, fill=FK_FILL, name='ruler')
    fig.emoji('💧', rx + 8, ry - 6, size=8)
    name_class_tag(fig, lx - 14, FK_Y + 27, *PARTS['ruler'])
    fig.text(lx - 14, FK_Y + 54, 'steers the run', **ROLE)
    xd, xu = PROP_X + 124, PROP_X + 164                          # the two arrows between the proposal and the ruler
    y0, y1 = ROW_Y + ROW_H + 2, FK_Y - 1
    fig.arrow([(xd, y0), (xd, y1)], head=3.0)
    fig.text(xd - 4, (y0 + y1) / 2 + 2.3, 'predicted clean graphs', justify='right', **NOTE)
    fig.arrow([(xu, y1), (xu, y0)], head=3.0)
    fig.badge(xu, (y0 + y1) / 2, 5.0, [('↻', 'mm')], size=6.5)
    fig.text(xu + 8, (y0 + y1) / 2 + 2.3, 'resample', **NOTE)

    # --- legend
    fig.legend(LEGEND_Y, [(*required_symbol(), 'required'), (*optional_symbol(), 'optional'),
                          (*target_symbol(), 'gets the target value')], x0=70, x1=370)
