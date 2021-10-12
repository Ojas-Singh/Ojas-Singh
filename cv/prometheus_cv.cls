\LoadClass{article}

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{prometheus_cv}[2020/10/14 Here to make editing your CV not an eternal torment]

\RequirePackage{titlesec}

% We customise the \section command 
\titleformat{\section}         
  {\Large\raggedright} % Make the \section headers large, small caps and left aligned
  {}{0.0em}                      % No label (prefix) and no horizontal separation between label and title body
  {}                           		% Code preceeding the title 
%   [{\titlerule[2pt]}]s
  [\color{coralred}{\titlerule[2.2pt]}]
%   \noindent
% {\color{Red} \rule{16cm}{0.5mm} }
  % Code following the title body. Here a horizontal line.
 

 
 %%% define the \datedsubsection command. %%%
 %%% The argument order follows the questions %%%
 %%% When? - Where? - What? - Details? %%%
 \newcommand{\datedsubsection}[4]{%	
 \cvsubsection[]{#3  \begin{minipage}{15cm}
	\raggedright
	%  \vspace*{+.2ex}\hspace*{+12.8cm} \footnotesize{#2}
	\hfill\textrf\footnotesize{#2}\raggedright
 \end{minipage}}%
 \begin{minipage}{5cm}
	\raggedright
	 \vspace*{+4.2ex}\hspace*{-2.5cm} \footnotesize{#1}\raggedright
 \end{minipage}
 \raggedright
	 
 \begin{ok}
 \hspace{0.5em}\small{#4}
 \end{ok}
 
}

\newcommand{\datedsubsectionnn}[4]{%	
 \cvsubsection[]{#3  \begin{minipage}{15cm}
	\raggedright
	%  \vspace*{+.2ex}\hspace*{+12.8cm} \footnotesize{#2}
	\hfill\textrf\footnotesize{#2}\raggedright
 \end{minipage}}%
 \begin{minipage}{5cm}
	\raggedright
	 \vspace*{+1.0ex}\hspace*{-2.5cm} \footnotesize{#1}\raggedright
 \end{minipage}
 \raggedright
	 
 \begin{ok}
 \hspace{0.5em}\small{#4}
 \end{ok}
 
}



%%% define the \datedsubsectionnarrow command %%%
% \newcommand{\datedsubsectionnarrow}[4]{%	
%  \cvsubsectionnarrow[#2]{#3 \hfill \small{\textlf{#2}}}%
%  \begin{minipage}{5cm}
% 	 \vspace*{-.7ex}\hspace*{+6cm}\footnotesize #1
%  \end{minipage}
%  \raggedright
 
%  \begin{normal}
% 	 \hspace{3.0em}{#4}
%  \end{normal}
% }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DEFINE OUR OWN SUBSECTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{cvsubsection}
\titleclass{\cvsubsection}{straight}[\part]
\titleformat{\cvsubsection}         
 {\raggedright} 
 {}{0.0em}                      
 {}                           
\titlespacing*{\cvsubsection}{0pt}{3.5ex plus 0.0ex minus .2ex}{-6.0ex}

% % The same but in narrow 
% \newcounter{cvsubsectionnarrow}
% \titleclass{\cvsubsectionnarrow}{straight}[\part]
% \titleformat{\cvsubsectionnarrow}         
%  {\raggedright} 
%  {}{0.0em}                      
%  {}                           
% \titlespacing*{\cvsubsectionnarrow}{0pt}{0pt}{-2.5ex}

\newcommand*{\customcventry}[7][.25em]{
  \begin{tabular}{@{}l} 
    {\bfseries #4}
  \end{tabular}
  \hfill% move it to the right
  \begin{tabular}{l@{}}
     {\bfseries #5}
  \end{tabular} \\
  \begin{tabular}{@{}l} 
    {\itshape #3}
  \end{tabular}
  \hfill% move it to the right
  \begin{tabular}{l@{}}
     {\itshape #2}
  \end{tabular}
  \ifx&#7&%
  \else{\\%
    \begin{minipage}{\maincolumnwidth}%
      \small#7%
    \end{minipage}}\fi%
  \par\addvspace{#1}}