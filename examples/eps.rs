use rasterize::*;
use std::{
    collections::HashMap, io::Read, num::ParseFloatError, str::FromStr, string::FromUtf8Error,
};

struct PSState {
    path: PathBuilder,
    stack: Vec<PSValue>,
    dict: HashMap<PSSymbol, PSValue>,
}

impl PSState {
    fn new() -> Self {
        Self {
            path: PathBuilder::new(),
            stack: Vec::new(),
            dict: HashMap::new(),
        }
    }

    fn pop(&mut self) -> Result<PSValue, PSValueError> {
        self.stack.pop().ok_or(PSValueError::StackEmpty)
    }

    fn eval(&mut self, cmd: PSValue) -> Result<(), PSValueError> {
        match cmd {
            PSValue::Symbol(symbol) => match symbol.as_str() {
                "def" => {
                    let value = self.pop()?;
                    let name = self.pop()?.to_quote()?;
                    self.dict.insert(name, value);
                }
                "dup" => {
                    let value = self.stack.last().ok_or(PSValueError::StackEmpty)?.clone();
                    self.stack.push(value);
                }
                _ => panic!(),
            },
            PSValue::Comment(_) => {}
            _ => self.stack.push(cmd),
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PSSymbol(String);

impl PSSymbol {
    fn new(string: String) -> Self {
        Self(string)
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone)]
enum PSValue {
    Number(Scalar),
    Symbol(PSSymbol),
    Quote(PSSymbol),
    String(String),
    Comment(String),
    Block(Vec<PSValue>),
    Array(Vec<PSValue>),
}

impl PSValue {
    fn to_quote(self) -> Result<PSSymbol, PSValueError> {
        match self {
            PSValue::Quote(symbol) => Ok(symbol),
            _ => Err(PSValueError::ExpectedQuote),
        }
    }
}

struct PSCmdParser<I> {
    input: I,
    buf: Vec<u8>,
}

impl<I: Read> PSCmdParser<I> {
    fn new(input: I) -> Self {
        Self {
            input,
            buf: Default::default(),
        }
    }

    fn parse(&mut self) -> Result<PSValue, PSValueError> {
        let byte = loop {
            let byte = self.read_byte()?;
            if !matches!(byte, b' ' | b'\n' | b'\r' | b'\t' | b'\x00' | b'\x0c') {
                break byte;
            }
        };

        match byte {
            b'/' => {
                let symbol = PSSymbol::new(self.read_symbol()?);
                Ok(PSValue::Quote(symbol))
            }
            b'%' => Ok(PSValue::Comment(self.read_while(|b| b != b'\n')?)),
            b'-' | b'+' | b'0'..=b'9' | b'.' => {
                self.push_byte(byte);
                let number = self.read_number()?;
                Ok(PSValue::Number(number))
            }
            b'_' | b'a'..=b'z' | b'A'..=b'Z' => {
                self.push_byte(byte);
                let symbol = PSSymbol::new(self.read_symbol()?);
                Ok(PSValue::Symbol(symbol))
            }
            b'(' => {
                let string = self.read_while(|byte| byte != b')')?;
                self.read_byte()?;
                Ok(PSValue::String(string))
            }
            b'{' => {
                let mut block = Vec::new();
                loop {
                    match self.parse() {
                        Ok(cmd) => block.push(cmd),
                        Err(PSValueError::UnclosedBlock) => break,
                        Err(error) => return Err(error),
                    }
                }
                Ok(PSValue::Block(block))
            }
            b'}' => Err(PSValueError::UnclosedBlock),
            b'[' => {
                let mut array = Vec::new();
                loop {
                    match self.parse() {
                        Ok(cmd) => array.push(cmd),
                        Err(PSValueError::UnclosedArray) => break,
                        Err(error) => return Err(error),
                    }
                }
                Ok(PSValue::Array(array))
            }
            b']' => Err(PSValueError::UnclosedArray),
            _ => Err(PSValueError::InputUnexpected(byte)),
        }
    }

    fn read_byte(&mut self) -> Result<u8, PSValueError> {
        match self.buf.pop() {
            None => {
                let mut byte = [0; 1];
                if let Err(error) = self.input.read_exact(&mut byte[..]) {
                    if error.kind() == std::io::ErrorKind::UnexpectedEof {
                        return Err(PSValueError::InputEmpty);
                    }
                    return Err(error.into());
                }
                Ok(byte[0])
            }
            Some(byte) => Ok(byte),
        }
    }

    fn push_byte(&mut self, byte: u8) {
        self.buf.push(byte)
    }

    fn read_while(&mut self, mut pred: impl FnMut(u8) -> bool) -> Result<String, PSValueError> {
        let mut result = Vec::new();
        loop {
            match self.read_byte() {
                Ok(byte) => {
                    if pred(byte) {
                        result.push(byte);
                    } else {
                        self.push_byte(byte);
                        break;
                    }
                }
                Err(PSValueError::InputEmpty) => break,
                Err(error) => return Err(error),
            }
        }
        Ok(String::from_utf8(result)?)
    }

    fn read_number(&mut self) -> Result<Scalar, PSValueError> {
        let number =
            self.read_while(|byte| matches!(byte, b'0'..=b'9' | b'.' | b'+' | b'-' | b'e' | b'E'))?;
        Ok(Scalar::from_str(&number)?)
    }

    fn read_symbol(&mut self) -> Result<String, PSValueError> {
        let symbol = self.read_while(|byte| {
            !matches!(
                byte,
                b' ' | b'\n' | b'\r' | b'\t' | b'\x00' | b'\x0c' | b'[' | b']' | b'{' | b'}'
            )
        })?;
        Ok(symbol)
    }
}

#[derive(Debug)]
enum PSValueError {
    IOError(std::io::Error),
    Float(ParseFloatError),
    Utf8(FromUtf8Error),
    InputUnexpected(u8),
    InputEmpty,
    UnclosedArray,
    UnclosedBlock,
    ExpectedQuote,
    StackEmpty,
}

impl From<std::io::Error> for PSValueError {
    fn from(error: std::io::Error) -> Self {
        Self::IOError(error)
    }
}

impl From<FromUtf8Error> for PSValueError {
    fn from(error: FromUtf8Error) -> Self {
        Self::Utf8(error)
    }
}

impl From<ParseFloatError> for PSValueError {
    fn from(error: ParseFloatError) -> Self {
        Self::Float(error)
    }
}

fn main() -> Result<(), PSValueError> {
    let file = std::fs::File::open("/mnt/data/downloads/tiger.eps")?;
    let mut parser = PSCmdParser::new(file);
    loop {
        let cmd = parser.parse()?;
        println!("{:?}", cmd);
    }
}
