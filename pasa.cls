\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{pasa}[2017/04/13, v2.0]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass[twoside,twocolumn,10pt]{article}

\RequirePackage{amsmath,amssymb}
\RequirePackage{etoolbox}
\RequirePackage[style]{abstract}
\RequirePackage{enumitem}
\RequirePackage[explicit]{titlesec}
\RequirePackage[T1]{fontenc}
\RequirePackage[tracking=smallcaps]{microtype}
\RequirePackage{lmodern}
\RequirePackage{caption}
\RequirePackage{fancyhdr}

\RequirePackage[letterpaper,top=.5in,includehead,headsep=14pt,headheight=28pt,left=.75in,textwidth=41.5pc,textheight=56pc,columnsep=18pt,footskip=18pt]{geometry}

\renewcommand{\title}[2][]{%
  \gdef\@shorttitle{#1}
  \gdef\@title{#2}
}

\renewcommand{\author}[2][]{%
  \gdef\@shortauthor{#1}
  \gdef\@author{#2}
}

\newcommand{\affil}[1]{\par\noindent\affilfont#1\par}

\def\@shortauthor{}
\def\@shorttitle{}
\def\@jyear{}
\def\@jnum{}
\def\@jid{}
\def\@jtitle{Publications of the Astronomical Society of Australia}
\def\@copyrightstatement{}
\def\@doi{}

\newcommand{\doi}[1]{\gdef\@doi{#1}}
\newcommand{\jyear}[1]{\gdef\@jyear{#1}}
\newcommand{\jid}[1]{\gdef\@jid{#1}}
\newcommand{\jtitle}[1]{\gdef\@jtitle{#1}}
\newcommand{\copyrightstatement}[1]{\gdef\@copyrightstatement{#1}}

\RequirePackage{environ}
\NewEnviron{frontmatter}{\twocolumn[\BODY]\saythanks}

\def\titlefont{\fontsize{18}{22}\bfseries\selectfont\raggedright\mathversion{normal}}
\def\authorfont{\normalsize\rightskip5.5pc \mathversion{normal}}
\def\affilfont{\footnotesize\raggedright\mathversion{normal}}
\def\abstractfont{\small\rightskip5.5pc}
\def\keywordfont{\small\rightskip5.5pc}

%% Section
\def\sectionfont{\normalsize\bfseries\rightskip1pc plus1fill\mathversion{bold}}
\def\subsectionfont{\normalsize\bfseries\rightskip1pc plus1fill\mathversion{bold}}
\def\subsubsectionfont{\normalsize\itshape\rightskip1pc plus1fill\mathversion{normal}}
\def\paragraphfont{\normalsize\itshape\mathversion{normal}}%
\def\subparagraphfont{\normalsize}%
\def\subsubparagraphfont{\normalsize}%
\def\bibliofont{\small}
\def\tabnotefont{\footnotesize}

\renewcommand{\@maketitle}{%
  \rule{\linewidth}{.27pt}\vspace{11.5pt}
  {\titlefont\@title\par}
  \vskip43.7pt
  \rule{\linewidth}{.27pt}\vspace{11.5pt}
  {\authorfont\@author\par}
  \vskip12pt
}

\fancyhf{}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt}
\fancyhead[LE,RO]{\thepage}
\fancyhead[LO]{\itshape\@shorttitle}
\fancyhead[RE]{\itshape\@shortauthor}
\fancypagestyle{plain}{%
  \fancyhf{}
  \lhead[L]{\footnotesize%
    \emph{\@jtitle}\ifdefempty{\@jid}{}{\space(\@jid)}\\%
    \ifdefempty{\@copyrightstatement}{}{\@copyrightstatement.\\}%
    \ifdefempty{\@doi}{}{doi:\space\@doi.}%
  }
  \cfoot{\thepage}
}

\renewcommand{\abstitlestyle}[1]{\textbf{#1}}
\renewcommand{\abstractnamefont}{\bfseries}
\renewcommand{\abstracttextfont}{\abstractfont}
\setlength{\absleftindent}{0pt}
\setlength{\absparindent}{0pt}
\apptocmd{\abstract}{\noindent\ignorespaces}{}{}
\appto{\endabstract}{\vskip12pt}

\newenvironment{keywords}{%
  \noindent\keywordfont\textbf{Keywords: }%
}{\vskip27pt}

\captionsetup*[figure]{font=footnotesize,labelfont=bf,labelsep=period,skip=10pt}
\captionsetup*[table]{font=small,position=top,labelfont=bf,labelsep=space,skip=10pt}


\newenvironment{acknowledgements}{\section{Acknowledgements}\small}{}

\titleformat{\section}
  {\sectionfont}
  {\thesection}
  {0.5em}
  {\MakeUppercase{#1}}

\titleformat{\subsection}
  {\subsectionfont}
  {\thesubsection}
  {0.5em}
  {#1}

\titleformat{\subsubsection}
  {\subsubsectionfont}
  {\thesubsubsection}
  {0.5em}
  {#1}

\titleformat{\paragraph}[runin]
  {\paragraphfont}
  {\theparagraph}
  {0.5em}
  {#1}
  
\titlespacing*{\section}{0pt}{18pt}{6pt}
\titlespacing*{\subsection}{0pt}{18pt}{6pt}
\titlespacing*{\subsubsection}{0pt}{10pt}{2pt}
\titlespacing*{\paragraph}{0pt}{6pt}{.5em}


\apptocmd{\quotation}{\setlength{\parindent}{0pt}%
\noindent\ignorespaces}{}{}
\patchcmd{\quotation}{\rightmargin\leftmargin}{\rightmargin0pt}{}{}
\setlist{noitemsep}
\newlist{unnumlist}{itemize}{1}
\setlist[unnumlist]{label={},leftmargin=2em,itemindent=-1em}

\def\x{@{\extracolsep{\fill}}}

\newcommand{\tabnote}[1]{\par\vskip1pt{\raggedright\tabnotefont #1\par}}

\RequirePackage{natbib}
\setlength{\bibsep}{0.3mm}

\frenchspacing

\endinput