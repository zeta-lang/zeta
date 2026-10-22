use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceSpan<'a> {
    pub file_name: &'a str,

    pub line: usize,
    pub column: usize,

    pub end_line: usize,
    pub end_column: usize,
}

impl Default for SourceSpan<'_> {
    fn default() -> Self {
        Self::nothing()
    }
}

impl<'a> SourceSpan<'a> {
    pub const fn nothing() -> Self {
        Self {
            file_name: "Unknown",
            line: 0,
            column: 0,
            end_line: 0,
            end_column: 0,
        }
    }

    pub fn new(file_name: &'a str, line: usize, column: usize) -> Self {
        Self {
            file_name,
            line,
            column,
            end_line: line,
            end_column: column,
        }
    }

    /// Combine two spans into one location for diagnostics spanning a
    /// multi-token construct. Since `SourceSpan` is a single point rather
    /// than a range, this keeps the earlier (start) position
    pub fn merge(self, other: SourceSpan<'a>) -> SourceSpan<'a> {
        let (start, end) = if (self.line, self.column) <= (other.line, other.column) {
            (self, other)
        } else {
            (other, self)
        };

        SourceSpan {
            file_name: self.file_name,
            line: start.line,
            column: start.column,
            end_line: end.line,
            end_column: end.column,
        }
    }
}

impl<'a> fmt::Display for SourceSpan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "line number {} at column {} inside of file named {}",
            self.line, self.column, self.file_name
        )
    }
}
